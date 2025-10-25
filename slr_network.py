import pdb
import copy
import utils
import torch
import types
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from modules.criterions import SeqKD
from modules import BiLSTMLayer, TemporalConv
import modules.resnet as resnet


####################################
from modules.tmm import MotionCue3D, TemporalMotionMix
####################################

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class NormLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(NormLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        outputs = torch.matmul(x, F.normalize(self.weight, dim=0))
        return outputs


class SLRModel(nn.Module):
    def __init__(
        self, num_classes, c2d_type, conv_type, use_bn=False,
        hidden_size=1024, gloss_dict=None, loss_weights=None,
        weight_norm=True, share_classifier=True, use_graph=False,
        # --- TMM knobs ---
        enable_tmm=False,
        tmm_location="pre_bilstm",
        tmm_alpha=1.0,                 # initial alpha (warm-start at identity)
        tmm_alpha_final=0.2,           # target alpha after warm-up
        tmm_alpha_warmup_epochs=10,    # epochs to anneal alpha
        enable_motion=True,
        # motion cue 3D stem
        motion3d_enable=True,
        motion3d_base_channels=32,
        motion3d_blocks=2,
        # gate → blank coupling
        gate_tau=0.7,                  # threshold for "high-g"
        gate_soft=0.1,                 # sigmoid softness
        gate_usage_prior=0.15,         # ρ in L_budget
        # misc
        blank_index=0,
    ):
        super(SLRModel, self).__init__()
        print(f"SLRModel: conv2d={c2d_type}, conv1d={conv_type}, use_bn={use_bn}, use_graph={use_graph}")
        self.decoder = None
        self.loss = dict()
        self.criterion_init()
        self.num_classes = num_classes
        self.loss_weights = loss_weights if loss_weights is not None else {}
        self.blank_index = int(blank_index)

        # 2D spatial backbone
        self.conv2d = getattr(resnet, c2d_type)(use_graph=use_graph)
        self.conv2d.fc = Identity()

        # temporal conv head
        self.conv1d = TemporalConv(
            input_size=512,
            hidden_size=hidden_size,
            conv_type=conv_type,
            use_bn=use_bn,
            num_classes=num_classes
        )

        # decoder
        self.decoder = utils.Decode(gloss_dict, num_classes, 'beam')

        # sequence model
        self.temporal_model = BiLSTMLayer(
            rnn_type='LSTM',
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True
        )

        # classifiers
        if weight_norm:
            self.classifier = NormLinear(hidden_size, self.num_classes)
            self.conv1d.fc = NormLinear(hidden_size, self.num_classes)
        else:
            self.classifier = nn.Linear(hidden_size, self.num_classes)
            self.conv1d.fc = nn.Linear(hidden_size, self.num_classes)
        if share_classifier:
            self.conv1d.fc = self.classifier

        # ---------------- TMM ----------------
        self.enable_tmm = bool(enable_tmm)
        self.tmm_location = str(tmm_location).lower()
        self.enable_motion = bool(enable_motion)

        self.gate_tau = float(gate_tau)
        self.gate_soft = float(gate_soft)
        self.gate_usage_prior = float(gate_usage_prior)

        if self.enable_tmm and self.enable_motion:
            # use 3D cue by default
            if motion3d_enable:
                self.motion_encoder = MotionCue3D(
                    in_ch=3,
                    base_channels=int(motion3d_base_channels),
                    num_blocks=int(motion3d_blocks),
                    feat_dim=hidden_size
                )
            else:
                # fallback: trivial motion (not recommended for locked run)
                from modules.tmm import MotionDiffEncoder  # only if present
                self.motion_encoder = MotionDiffEncoder(in_ch=3, feat_dim=hidden_size)
        else:
            self.motion_encoder = None

        if self.enable_tmm:
            self.tmm = TemporalMotionMix(d=hidden_size, alpha=float(tmm_alpha))
            # store anneal targets
            self._alpha_init  = float(tmm_alpha)
            self._alpha_final = float(tmm_alpha_final)
            self._alpha_warm  = int(tmm_alpha_warmup_epochs)
        else:
            self.tmm = None

        print(
            f"SLRModel: enable_tmm={self.enable_tmm}, tmm_location={self.tmm_location}, "
            f"enable_motion={self.enable_motion}, motion3d_enable={motion3d_enable}"
        )


    ################################################

    def _align_motion_to_feat(self, motion_B_T_D, feat_len, total_T):
        """
        motion_B_T_D: (B, T_raw, D)
        feat_len: (B,) after TCN
        Returns (T’, B, D)
        """
        B, T_raw, D = motion_B_T_D.shape
        T_new = int(feat_len[0].item())
        idx = torch.linspace(0, T_raw - 1, steps=T_new, device=motion_B_T_D.device).round().long()
        m = motion_B_T_D[:, idx]            # (B, T’, D)
        m = m.permute(1, 0, 2).contiguous() # (T’, B, D)
        return m

    @torch.no_grad()
    def set_alpha_by_epoch(self, epoch: int):
        if not (self.enable_tmm and self.tmm is not None):
            return
        if self._alpha_warm <= 0:
            a = self._alpha_final
        else:
            t = max(0, min(epoch, self._alpha_warm))
            a = self._alpha_init + (self._alpha_final - self._alpha_init) * (t / self._alpha_warm)
        self.tmm.set_alpha(a)
    ################################################

    def backward_hook(self, module, grad_input, grad_output):
        for g in grad_input:
            g[g != g] = 0

    def masked_bn(self, inputs, len_x):
        def pad(tensor, length):
            return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_()])

        x = torch.cat([inputs[len_x[0] * idx:len_x[0] * idx + lgt] for idx, lgt in enumerate(len_x)])
        x = self.conv2d(x)
        x = torch.cat([pad(x[sum(len_x[:idx]):sum(len_x[:idx + 1])], len_x[0])
                       for idx, lgt in enumerate(len_x)])
        return x

    # def forward(self, x, len_x, label=None, label_lgt=None):

    #     if len(x.shape) == 5:
    #         # videos
    #         batch, temp, channel, height, width = x.shape
    #         framewise = self.conv2d(x.permute(0,2,1,3,4)).view(batch, temp, -1).permute(0,2,1) # btc -> bct # (B, T, 512) -> (B, 512, T)
    #         ############################################
    #         # prepare motion if needed
    #         if self.enable_tmm and self.enable_motion:
    #             # MotionDiffEncoder expects (B, T, C, H, W)
    #             # with torch.no_grad(): # motion is a cue; you can also learn end-to-end by removing no_grad
    #             #     m_raw = self.motion_encoder(x) # (B, T, d)
    #             m_raw = self.motion_encoder(x) # (B, T, d)
    #         else:
    #             m_raw = None
    #         ############################################
    #     else:
    #         # features path (already (B, 512, T))
    #         framewise = x
    #         m_raw = None # cannot compute motion without raw frames


    #     # TemporalConv
    #     conv1d_outputs = self.conv1d(framewise, len_x) # visual_feat: (T', B, d), conv_logits: (T', B, num_class), feat_len: (B,)

    #     ############################################
    #     # x: T', B, d
    #     # x = conv1d_outputs['visual_feat']
    #     z_tcn = conv1d_outputs['visual_feat'] # (T', B, d)
    #     ############################################

        
    #     lgt = conv1d_outputs['feat_len'].cpu() # (B,)

    #     ############################################
    #     # ------- TMM location: pre_bilstm -------
    #     if self.enable_tmm and self.tmm_location == "pre_bilstm":
    #         if m_raw is not None:
    #             m_aligned = self._align_motion_to_feat(m_raw, lgt, temp) # (T', B, d)
    #         else:
    #             # fallback: zero motion
    #             m_aligned = torch.zeros_like(z_tcn)
    #         z_clean, g, g_pre_sigmoid = self.tmm(z_tcn, m_aligned) # (T', B, d) , (T', B, 1)
    #         z_for_lstm = z_clean
    #     else:
    #         z_for_lstm = z_tcn

    #     # ------------ BiLSTM ------------
    #     # tm_outputs = self.temporal_model(x, lgt)
    #     tm_outputs = self.temporal_model(z_for_lstm, lgt)
    #     z_seq = tm_outputs['predictions']           # (T’, B, H)


    #     ############################################
    #     # ---- TMM location: post_bilstm ----
    #     if self.enable_tmm and self.tmm_location == 'post_bilstm':
    #         if m_raw is not None:
    #             m_aligned = self._align_motion_to_feat(m_raw, lgt, temp)
    #         else:
    #             m_aligned = torch.zeros_like(z_seq)
    #         z_clean, g = self.tmm(z_seq, m_aligned)
    #         z_out = z_clean
    #     else:
    #         z_out = z_seq

    #     # ---- Classifiers ----
    #     seq_logits = self.classifier(z_out)                     # (T’, B, C)
    #     conv_logits = conv1d_outputs['conv_logits']             # (T’, B, C)

    #     # outputs = self.classifier(tm_outputs['predictions'])
    #     # outputs = self.classifier(tm_outputs['predictions'])


    #     pred = None if self.training \
    #         else self.decoder.decode(seq_logits, lgt, batch_first=False, probs=False)
    #     conv_pred = None if self.training \
    #         else self.decoder.decode(conv_logits, lgt, batch_first=False, probs=False)
        

    #     return {
    #         "framewise_features": framewise,
    #         "visual_features": z_tcn,
    #         "temproal_features": z_seq,
    #         "feat_len": lgt,
    #         "conv_logits": conv_logits,
    #         "sequence_logits": seq_logits,
    #         "conv_sents": conv_pred,
    #         "recognized_sents": pred,
    #         # OPTIONAL: expose gate for diagnostics
    #         "tmm_gate": g if (self.enable_tmm) else None,
    #         "tmm_gate_pre": g_pre_sigmoid if (self.enable_tmm) else None,
    #     }


    # --------------- forward ---------------

    def forward(self, x, len_x, label=None, label_lgt=None):
        """
        x: (B, T, 3, H, W) for video mode or (B, 512, T) for feature mode
        """
        g = None
        g_pre = None
        m_raw = None
        m_aligned = None
        phase_energy = None  # (T', B)

        if len(x.shape) == 5:
            # videos: (B, T, C, H, W)
            B, T, C, H, W = x.shape
            framewise = self.conv2d(x.permute(0, 2, 1, 3, 4)).view(B, T, -1).permute(0, 2, 1)  # (B, 512, T)
            if self.enable_tmm and self.enable_motion and (self.motion_encoder is not None):
                # learnable 3D cue
                m_raw = self.motion_encoder(x)  # (B, T, d)
        else:
            # pre-extracted features: (B, 512, T)
            framewise = x
            m_raw = None

        # TCN
        conv1d_outputs = self.conv1d(framewise, len_x)
        z_tcn = conv1d_outputs['visual_feat']   # (T', B, d)
        lgt   = conv1d_outputs['feat_len'].cpu()

        # Prepare motion aligned to T'
        if self.enable_tmm and (m_raw is not None):
            m_aligned = self._align_motion_to_feat(m_raw, lgt, total_T=None)  # (T', B, d)
            # phase energy from cue diffs (unsupervised target), shape: (T', B)
            with torch.no_grad():
                # ||m_t - m_{t-1}||_2 over d
                diff = torch.zeros_like(m_aligned[:, :, 0])
                if m_aligned.size(0) > 1:
                    dlt = m_aligned[1:] - m_aligned[:-1]
                    diff[1:] = torch.linalg.vector_norm(dlt, dim=-1)
                # per-sequence min-max normalize to [0,1]
                mn = diff.min(dim=0, keepdim=True)[0]
                mx = diff.max(dim=0, keepdim=True)[0]
                phase_energy = (diff - mn) / (mx - mn + 1e-6)  # (T', B)

        # ---- TMM ----
        if self.enable_tmm and (self.tmm_location == "pre_bilstm"):
            if m_aligned is None:
                m_aligned = torch.zeros_like(z_tcn)
            z_clean, g, g_pre = self.tmm(z_tcn, m_aligned)  # (T', B, d), (T', B, d), (T', B, d)
            z_for_lstm = z_clean
        else:
            z_for_lstm = z_tcn

        # BiLSTM
        tm_outputs = self.temporal_model(z_for_lstm, lgt)
        z_seq = tm_outputs['predictions']  # (T’, B, d)

        # post-LSTM TMM (not used in locked plan but kept for compatibility)
        if self.enable_tmm and (self.tmm_location == "post_bilstm"):
            if m_aligned is None:
                m_aligned = torch.zeros_like(z_seq)
            z_out, g, g_pre = self.tmm(z_seq, m_aligned)
        else:
            z_out = z_seq

        # classifiers
        seq_logits  = self.classifier(z_out)                # (T’, B, C)
        conv_logits = conv1d_outputs['conv_logits']         # (T’, B, C)

        pred = None if self.training else self.decoder.decode(seq_logits, lgt, batch_first=False, probs=False)
        conv_pred = None if self.training else self.decoder.decode(conv_logits, lgt, batch_first=False, probs=False)

        return {
            "framewise_features": framewise,
            "visual_features": z_tcn,
            "temproal_features": z_seq,
            "feat_len": lgt,
            "conv_logits": conv_logits,
            "sequence_logits": seq_logits,
            "conv_sents": conv_pred,
            "recognized_sents": pred,
            "tmm_gate": g if self.enable_tmm else None,              # (T', B, d) or None
            "tmm_gate_pre": g_pre if self.enable_tmm else None,      # (T', B, d) or None
            "tmm_phase_energy": phase_energy,                        # (T', B) or None
        }
    

    # def criterion_calculation(self, ret_dict, label, label_lgt):
    #     loss = 0
    #     for k, weight in self.loss_weights.items():
    #         if k == 'ConvCTC':
    #             loss += weight * self.loss['CTCLoss'](ret_dict["conv_logits"].log_softmax(-1),
    #                                                   label.cpu().int(), ret_dict["feat_len"].cpu().int(),
    #                                                   label_lgt.cpu().int()).mean()
    #         elif k == 'SeqCTC':
    #             loss += weight * self.loss['CTCLoss'](ret_dict["sequence_logits"].log_softmax(-1),
    #                                                   label.cpu().int(), ret_dict["feat_len"].cpu().int(),
    #                                                   label_lgt.cpu().int()).mean()
    #         elif k == 'Dist':
    #             loss += weight * self.loss['distillation'](ret_dict["conv_logits"],
    #                                                        ret_dict["sequence_logits"].detach(),
    #                                                        use_blank=False)
    #     return loss

    # def criterion_init(self):
    #     self.loss['CTCLoss'] = torch.nn.CTCLoss(reduction='none', zero_infinity=False)
    #     self.loss['distillation'] = SeqKD(T=8)
    #     return self.loss

    # ---------------- losses ----------------

    def criterion_calculation(self, ret_dict, label, label_lgt):
        """
        Adds:
          - Blank guidance (if weight 'Blank' > 0)
          - Phase loss     (if weight 'Phase' > 0)
        """
        loss = 0.0
        w_seq   = float(self.loss_weights.get('SeqCTC', 1.0))
        w_conv  = float(self.loss_weights.get('ConvCTC', 1.0))
        w_kd    = float(self.loss_weights.get('Dist', 0.0))
        w_blank = float(self.loss_weights.get('Blank', 0.0))
        w_phase = float(self.loss_weights.get('Phase', 0.0))

        # CTC losses
        if w_conv != 0.0:
            loss += w_conv * self.loss['CTCLoss'](
                ret_dict["conv_logits"].log_softmax(-1),
                label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                label_lgt.cpu().int()
            ).mean()

        if w_seq != 0.0:
            loss += w_seq * self.loss['CTCLoss'](
                ret_dict["sequence_logits"].log_softmax(-1),
                label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                label_lgt.cpu().int()
            ).mean()

        # KD
        if w_kd != 0.0:
            loss += w_kd * self.loss['distillation'](
                ret_dict["conv_logits"], ret_dict["sequence_logits"].detach(), use_blank=False
            )

        # ---- Blank guidance (affect classifier, NOT gate) ----
        if (w_blank != 0.0) and (ret_dict.get("tmm_gate", None) is not None):
            g = ret_dict["tmm_gate"].float()                    # (T', B, d)
            g_scalar = g.mean(dim=-1)                           # (T', B)
            # soft mask: high when gate is high
            w = torch.sigmoid((g_scalar - self.gate_tau) / self.gate_soft).detach()  # stop-grad
            log_probs = F.log_softmax(ret_dict["sequence_logits"], dim=-1)           # (T', B, C)
            log_p_blank = log_probs[..., self.blank_index]                            # (T', B)
            L_blank = -(w * log_p_blank).mean()
            loss += w_blank * L_blank

        # ---- Phase loss (teach the gate): L_align + L_budget ----
        if (w_phase != 0.0) and (ret_dict.get("tmm_gate", None) is not None):
            g = ret_dict["tmm_gate"].float()             # (T', B, d)
            g_scalar = g.mean(dim=-1)                    # (T', B)

            L_align = 0.0
            E = ret_dict.get("tmm_phase_energy", None)   # (T', B) or None

            if E is not None:
                # MSE between gate and normalized cue energy
                L_align = F.mse_loss(g_scalar, E)

            # budget prior (fraction of high gate)
            mean_g = g_scalar.mean()                     # scalar
            L_budget = (mean_g - self.gate_usage_prior) ** 2

            L_phase = L_align + 0.05 * L_budget         # 0.05 inside to keep 'Phase' scale sane
            loss += w_phase * L_phase

        return loss

    def criterion_init(self):
        self.loss['CTCLoss'] = torch.nn.CTCLoss(reduction='none', zero_infinity=False)
        self.loss['distillation'] = SeqKD(T=8)
        return self.loss
