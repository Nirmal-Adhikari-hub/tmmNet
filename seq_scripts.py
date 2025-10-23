import os
import pdb
import sys
import copy
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler
from einops import rearrange
from collections import defaultdict
from utils.metrics import wer_list
from utils.misc import *


# ---------- small helpers ----------
def _choose_amp_dtype():
    """
    Prefer bf16 on A100/modern GPUs (safe and stable for eval/forward);
    fall back to fp16 elsewhere.
    """
    if torch.cuda.is_available() and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
        print("[Using bf16 autocast]")
        return torch.bfloat16
    return torch.float16


def _safe_logit(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Compute pre-sigmoid logits from probabilities g in (0,1):
        logit(p) = log(p) - log(1-p)
    with clamping for numerical safety. Shape is preserved.
    """
    p = p.clamp(min=eps, max=1.0 - eps)
    return torch.log(p) - torch.log1p(-p)
# -----------------------------------


def seq_train(loader, model, optimizer, device, epoch_idx, recoder):
    model.train()
    loss_value = []
    clr = [group['lr'] for group in optimizer.optimizer.param_groups]

    amp_dtype = _choose_amp_dtype()
    use_scaler = (amp_dtype == torch.float16)
    scaler = GradScaler(enabled=use_scaler)

    global_step_base = epoch_idx * len(loader)
    pbar = tqdm(loader, total=len(loader), dynamic_ncols=True, leave=True, disable=not is_main_process())

    for batch_idx, data in enumerate(pbar):
        vid = device.data_to_device(data[0])
        vid_lgt = device.data_to_device(data[1])
        label = device.data_to_device(data[2])
        label_lgt = device.data_to_device(data[3])

        optimizer.zero_grad()
        with autocast(dtype=amp_dtype):
            ret_dict = model(vid, vid_lgt, label=label, label_lgt=label_lgt)
            if len(device.gpu_list) > 1:
                loss = model.module.criterion_calculation(ret_dict, label, label_lgt)
            else:
                loss = model.criterion_calculation(ret_dict, label, label_lgt)

        if np.isinf(loss.item()) or np.isnan(loss.item()):
            print('loss is nan')
            print(str(data[1]) + '  frames', str(data[3]) + '  glosses')
            continue

        if use_scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer.optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.optimizer.step()

        if len(device.gpu_list) > 1:
            torch.cuda.synchronize()
            torch.distributed.reduce(loss, dst=0)

        loss_value.append(loss.item())
        step = global_step_base + batch_idx

        if batch_idx % recoder.log_interval == 0 and is_main_process():
            # ---- TMM diagnostics ----
            g = ret_dict.get("tmm_gate", None)              # post-sigmoid (T', B, 1 or d)
            gp = ret_dict.get("tmm_gate_pre", None)         # pre-sigmoid if model exposes it
            if g is not None:
                g = g.float()
                gm = g.mean().item()
                gs = g.std().item()
                g50 = (g > 0.5).float().mean().item()

                # derive pre-sigmoid if not provided
                if gp is None:
                    gp = _safe_logit(g)
                gp = gp.float()
                gp_mean = gp.mean().item()

                recoder.log_metrics({
                    "epoch": epoch_idx,
                    "train/tmm_gate_mean": float(gm),
                    "train/tmm_gate_std": float(gs),
                    "train/tmm_gate_frac>0.5": float(g50),
                    "train/tmm_gate_pre_mean": float(gp_mean),   # NEW: pre-sigmoid mean
                }, step=step)

            # ---- standard progress ----
            pbar.set_description(f"Epoch {epoch_idx}")
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{clr[0]:.6f}")

            recoder.log_metrics({
                "epoch": epoch_idx,
                "train/ctc_loss": float(loss.item()),
                "train/lr": float(clr[0]),
            }, step=step)

        del ret_dict, loss, vid, vid_lgt, label, label_lgt

    optimizer.scheduler.step()
    if is_main_process():
        pbar.close()
        recoder.print_log('\tMean training loss: {:.10f}.'.format(np.mean(loss_value)))
        recoder.log_metrics({
            "epoch": epoch_idx,
            "train_epoch/ctc_loss": float(np.mean(loss_value)),
        })
    return


def seq_eval(cfg, loader, model, device, mode, epoch, work_dir, recoder, evaluate_tool="python"):
    """
    Evaluation forward is wrapped in AMP with bf16 where supported.
    Decoding and metrics remain exactly the same. Data order is preserved by the DataLoader.
    """
    model.eval()
    results = defaultdict(dict)

    # post-sigmoid accumulators
    gate_sum = 0.0
    gate_sqsum = 0.0
    gate_cnt = 0
    gate_above = 0.0

    # pre-sigmoid accumulator (mean only)
    gate_pre_sum = 0.0

    amp_dtype = _choose_amp_dtype()
    select_metric = str(getattr(cfg, "select_metric", "best")).lower()  # "lstm", "conv", or "best"

    for batch_idx, data in enumerate(tqdm(loader, disable=not is_main_process())):
        recoder.record_timer("device")
        vid = device.data_to_device(data[0])
        vid_lgt = device.data_to_device(data[1])
        label = device.data_to_device(data[2])
        label_lgt = device.data_to_device(data[3])
        info = [d['fileid'] for d in data[-1]]
        gloss = [d['label'] for d in data[-1]]

        with torch.inference_mode():
            with autocast(dtype=amp_dtype):
                ret_dict = model(vid, vid_lgt, label=label, label_lgt=label_lgt)

            # ----- TMM stats -----
            g = ret_dict.get("tmm_gate", None)              # post-sigmoid
            gp = ret_dict.get("tmm_gate_pre", None)         # pre-sigmoid (optional)

            if g is not None:
                g = g.float()
                gate_sum   += g.sum().item()
                gate_sqsum += (g * g).sum().item()
                gate_cnt   += g.numel()
                gate_above += (g > 0.5).float().sum().item()

                # compute/accumulate pre-sigmoid mean
                if gp is None:
                    gp = _safe_logit(g)
                gate_pre_sum += gp.float().sum().item()

            # ----- collect hypotheses -----
            for inf, conv_sents, recognized_sents, gl in zip(
                info, ret_dict['conv_sents'], ret_dict['recognized_sents'], gloss
            ):
                results[inf]['conv_sents'] = conv_sents
                results[inf]['recognized_sents'] = recognized_sents
                results[inf]['gloss'] = gl

        del vid, vid_lgt, label, label_lgt, ret_dict

    # compute WERs
    gls_hyp = [' '.join(results[n]['conv_sents']) for n in results]
    gls_ref = [results[n]['gloss'] for n in results]
    wer_results_con = wer_list(hypotheses=gls_hyp, references=gls_ref)

    gls_hyp = [' '.join(results[n]['recognized_sents']) for n in results]
    wer_results = wer_list(hypotheses=gls_hyp, references=gls_ref)

    # selection policy
    if select_metric == "lstm":
        reg_per = wer_results
    elif select_metric == "conv":
        reg_per = wer_results_con
    else:
        reg_per = wer_results if wer_results['wer'] < wer_results_con['wer'] else wer_results_con

    # log to txt (has sub/ins/del if your wer_list returns them)
    recoder.print_log(
        f"\tEpoch: {epoch} {mode} CONV  wer:{wer_results_con['wer']:.4f} "
        f"sub:{wer_results_con.get('sub', 0.0):.4f} ins:{wer_results_con['ins']:.4f} del:{wer_results_con['del']:.4f}",
        f"{work_dir}/{mode}.txt"
    )
    recoder.print_log(
        f"\tEpoch: {epoch} {mode} LSTM  wer:{wer_results['wer']:.4f} "
        f"sub:{wer_results.get('sub', 0.0):.4f} ins:{wer_results['ins']:.4f} del:{wer_results['del']:.4f}",
        f"{work_dir}/{mode}.txt"
    )

    # aggregate gate stats
    if gate_cnt > 0:
        gate_mean = gate_sum / gate_cnt
        gate_var  = max(gate_sqsum / gate_cnt - gate_mean * gate_mean, 0.0)
        gate_std  = gate_var ** 0.5
        gate_frac = gate_above / gate_cnt
        gate_pre_mean = gate_pre_sum / gate_cnt
    else:
        gate_mean = gate_std = gate_frac = gate_pre_mean = float('nan')

    # wandb: DEV/TEST curves vs epoch
    if is_main_process():
        metrics = {
            "epoch": epoch,
            f"{mode}/WER_LSTM": float(wer_results['wer']),
            f"{mode}/WER_CONV": float(wer_results_con['wer']),
            f"{mode}/WER": float(reg_per['wer']),
            f"{mode}/INS": float(reg_per['ins']),
            f"{mode}/DEL": float(reg_per['del']),
        }
        if gate_cnt > 0:
            metrics.update({
                f"{mode}/tmm_gate_mean": float(gate_mean),           # post-sigmoid
                f"{mode}/tmm_gate_std": float(gate_std),
                f"{mode}/tmm_gate_frac>0.5": float(gate_frac),
                f"{mode}/tmm_gate_pre_mean": float(gate_pre_mean),   # NEW: pre-sigmoid mean
            })
        recoder.log_metrics(metrics)

    return {"wer": reg_per['wer'], "ins": reg_per['ins'], 'del': reg_per['del']}
