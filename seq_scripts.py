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



def seq_train(loader, model, optimizer, device, epoch_idx, recoder):
    model.train()
    # optimizer.scheduler.step(epoch_idx)
    loss_value = []
    clr = [group['lr'] for group in optimizer.optimizer.param_groups]
    scaler = GradScaler()

    # use a monotonically increasing global step for WANDB train curves
    global_step_base = epoch_idx * len(loader)

    pbar = tqdm(loader, total=len(loader), dynamic_ncols=True, leave=True, disable=not is_main_process())

    for batch_idx, data in enumerate(pbar):
        vid = device.data_to_device(data[0])
        vid_lgt = device.data_to_device(data[1])
        label = device.data_to_device(data[2])
        label_lgt = device.data_to_device(data[3])
        optimizer.zero_grad()
        with autocast():

            ret_dict = model(vid, vid_lgt, label=label, label_lgt=label_lgt)
            if len(device.gpu_list)>1:
                loss = model.module.criterion_calculation(ret_dict, label, label_lgt)
            else:
                loss = model.criterion_calculation(ret_dict, label, label_lgt)

        if np.isinf(loss.item()) or np.isnan(loss.item()):
            print('loss is nan')
            print(str(data[1])+'  frames', str(data[3])+'  glosses')
            continue
        scaler.scale(loss).backward()
        scaler.step(optimizer.optimizer)
        scaler.update() 
        if len(device.gpu_list)>1:
            torch.cuda.synchronize() 
            torch.distributed.reduce(loss, dst=0)

        loss_value.append(loss.item())
        step = global_step_base + batch_idx
        if batch_idx % recoder.log_interval == 0 and is_main_process():
            g = ret_dict.get("tmm_gate", None)
            if g is not None:
                # g: (T', B, 1) or (T', B, d). Flatten safely.
                gm = g.float().mean().item()
                gs = g.float().std().item()
                g50 = (g > 0.5).float().mean().item()
                recoder.log_metrics({
                    "epoch": epoch_idx,
                    "train/tmm_gate_mean": float(gm),
                    "train/tmm_gate_std": float(gs),
                    "train/tmm_gate_frac>0.5": float(g50),
                }, step=step)

            # recoder.print_log(
            #     '\tEpoch: {}, Batch({}/{}) done. Loss: {:.8f}  lr:{:.6f}'
            #         .format(epoch_idx, batch_idx, len(loader), loss.item(), clr[0]))

            pbar.set_description(f"Epoch {epoch_idx}")
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{clr[0]:.6f}")

            # wandb: TRAIN curves vs global step (not epoch)
            # step = global_step_base + batch_idx
            recoder.log_metrics({
                "epoch": epoch_idx,    # still useful for grouping
                "train/ctc_loss": float(loss.item()),
                "train/lr": float(clr[0]),
            }, step=step)

        del ret_dict
        del loss
    optimizer.scheduler.step()
    if is_main_process():
        pbar.close()
        recoder.print_log('\tMean training loss: {:.10f}.'.format(np.mean(loss_value)))
        # optional: log per-epoch averaged train loss at the epoch index
        recoder.log_metrics({
            "epoch": epoch_idx,
            "train_epoch/ctc_loss": float(np.mean(loss_value)),
        })
    return


def seq_eval(cfg, loader, model, device, mode, epoch, work_dir, recoder, evaluate_tool="python"):
    model.eval()
    results=defaultdict(dict)

    gate_sum = 0.0
    gate_sqsum = 0.0
    gate_cnt = 0
    gate_above = 0.0


    for batch_idx, data in enumerate(tqdm(loader, disable=not is_main_process())):
        recoder.record_timer("device")
        vid = device.data_to_device(data[0])
        vid_lgt = device.data_to_device(data[1])
        label = device.data_to_device(data[2])
        label_lgt = device.data_to_device(data[3])
        info = [d['fileid'] for d in data[-1]]
        gloss = [d['label'] for d in data[-1]]
        with torch.no_grad():
            ret_dict = model(vid, vid_lgt, label=label, label_lgt=label_lgt)


            g = ret_dict.get("tmm_gate", None)
            if g is not None:
                g = g.float()
                gate_sum   += g.sum().item()
                gate_sqsum += (g * g).sum().item()
                gate_cnt   += g.numel()
                gate_above += (g > 0.5).float().sum().item()



            for inf, conv_sents, recognized_sents, gl in zip(info, ret_dict['conv_sents'], ret_dict['recognized_sents'], gloss):
                results[inf]['conv_sents'] = conv_sents
                results[inf]['recognized_sents'] = recognized_sents
                results[inf]['gloss'] = gl
    gls_hyp = [' '.join(results[n]['conv_sents']) for n in results]
    gls_ref = [results[n]['gloss'] for n in results]
    wer_results_con = wer_list(hypotheses=gls_hyp, references=gls_ref)
    gls_hyp = [' '.join(results[n]['recognized_sents']) for n in results]
    wer_results = wer_list(hypotheses=gls_hyp, references=gls_ref)
    if wer_results['wer'] < wer_results_con['wer']:
        reg_per = wer_results
    else:
        reg_per = wer_results_con
    recoder.print_log('\tEpoch: {} {} done. Conv wer: {:.4f}  ins:{:.4f}, del:{:.4f}'.format(
        epoch, mode, wer_results_con['wer'], wer_results_con['ins'], wer_results_con['del']),
        f"{work_dir}/{mode}.txt")
    recoder.print_log('\tEpoch: {} {} done. LSTM wer: {:.4f}  ins:{:.4f}, del:{:.4f}'.format(
        epoch, mode, wer_results['wer'], wer_results['ins'], wer_results['del']), f"{work_dir}/{mode}.txt")
    
    if gate_cnt > 0:
        gate_mean = gate_sum / gate_cnt
        gate_var  = max(gate_sqsum / gate_cnt - gate_mean * gate_mean, 0.0)
        gate_std  = gate_var ** 0.5
        gate_frac = gate_above / gate_cnt
    else:
        gate_mean = gate_std = gate_frac = float('nan')

    
    # wandb: DEV/TEST curves vs epoch
    if is_main_process():
        metrics = {
            "epoch": epoch,
            f"{mode}/WER_LSTM": float(wer_results['wer']),
            f"{mode}/WER_CONV": float(wer_results_con['wer']),
            f"{mode}/WER": float(reg_per['wer']),                # best of the two, like your reg_per
            f"{mode}/INS": float(reg_per['ins']),
            f"{mode}/DEL": float(reg_per['del']),
        }

        # add TMM stats if present
        if gate_cnt > 0:
            metrics.update({
                f"{mode}/tmm_gate_mean": float(gate_mean),
                f"{mode}/tmm_gate_std": float(gate_std),
                f"{mode}/tmm_gate_frac>0.5": float(gate_frac),
            })

            
        recoder.log_metrics(metrics, step=epoch)

    return {"wer":reg_per['wer'], "ins":reg_per['ins'], 'del':reg_per['del']}
 
