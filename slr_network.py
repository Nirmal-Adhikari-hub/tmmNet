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
from modules.tmm import MotionDiffEncoder, TemporalMotionMix
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
            # NEW for TMM
            enable_tmm=False,
            tmm_location="post_bilstm",
            tmm_alpha=0.2,
            enable_motion=True,
    ):
        super(SLRModel, self).__init__()
        print(f"SLRModel: conv2d={c2d_type}, conv1d={conv_type}, use_bn={use_bn}, use_graph={use_graph}")
        self.decoder = None
        self.loss = dict()
        self.criterion_init()
        self.num_classes = num_classes
        self.loss_weights = loss_weights
        self.conv2d = getattr(resnet, c2d_type)(use_graph=use_graph)
        self.conv2d.fc = Identity()

        self.conv1d = TemporalConv(input_size=512,
                                   hidden_size=hidden_size,
                                   conv_type=conv_type,
                                   use_bn=use_bn,
                                   num_classes=num_classes)
        self.decoder = utils.Decode(gloss_dict, num_classes, 'beam')
        self.temporal_model = BiLSTMLayer(rnn_type='LSTM', input_size=hidden_size, hidden_size=hidden_size,
                                          num_layers=2, bidirectional=True)
        if weight_norm:
            self.classifier = NormLinear(hidden_size, self.num_classes)
            self.conv1d.fc = NormLinear(hidden_size, self.num_classes)
        else:
            self.classifier = nn.Linear(hidden_size, self.num_classes)
            self.conv1d.fc = nn.Linear(hidden_size, self.num_classes)
        if share_classifier:
            self.conv1d.fc = self.classifier

        ############################################
        self.enable_tmm = enable_tmm
        self.tmm_location = tmm_location
        self.enable_motion = enable_motion

        if self.enable_tmm:
            self.motion_encoder = MotionDiffEncoder(in_ch=3, feat_dim=hidden_size) if self.enable_motion else None
            self.tmm = TemporalMotionMix(d=hidden_size, alpha=tmm_alpha)
        else:
            self.motion_encoder = None
            self.tmm = None

        print(f"SLRModel: conv2d={c2d_type}, conv1d={conv_type}, use_bn={use_bn}, use_graph={use_graph}, "
                f"enable_tmm={self.enable_tmm}, tmm_location={self.tmm_location}, enable_motion={self.enable_motion}")

        ############################################


    ################################################
    def _align_motion_to_feat(self, motion_B_T_D, feat_len, total_T):
        """
        motion_B_T_D: (B, T_raw, D)
        feat_len: (B,) after TCN (all equal in this repo)
        total_T: int (raw T)
        Returns (T’, B, D) aligned to sequence feature length
        """
        B, T_raw, D = motion_B_T_D.shape
        T_new = int(feat_len[0].item())
        # simple uniform sampling
        idx = torch.linspace(0, T_raw-1, steps=T_new).round().long().to(motion_B_T_D.device)
        m = motion_B_T_D[:, idx]                    # (B, T’, D)
        m = m.permute(1, 0, 2).contiguous()         # (T’, B, D)
        return m
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

    def forward(self, x, len_x, label=None, label_lgt=None):

        if len(x.shape) == 5:
            # videos
            batch, temp, channel, height, width = x.shape
            framewise = self.conv2d(x.permute(0,2,1,3,4)).view(batch, temp, -1).permute(0,2,1) # btc -> bct # (B, 512, T) -> (B, 512, T)
            ############################################
            # prepare motion if needed
            if self.enable_tmm and self.enable_motion:
                # MotionDiffEncoder expects (B, T, C, H, W)
                with torch.no_grad(): # motion is a cue; you can also learn end-to-end by removing no_grad
                    m_raw = self.motion_encoder(x) # (B, T, hidden_size)
            else:
                m_raw = None
            ############################################
        else:
            # features path (already (B, 512, T))
            framewise = x
            m_raw = None # cannot compute motion without raw frames


        # TemporalConv
        conv1d_outputs = self.conv1d(framewise, len_x)

        ############################################
        # x: T, B, C
        # x = conv1d_outputs['visual_feat']
        z_tcn = conv1d_outputs['visual_feat'] # (T', B, H)
        ############################################

        
        lgt = conv1d_outputs['feat_len'].cpu() # (B,)

        ############################################
        # ------- TMM location: pre_bilstm -------
        if self.enable_tmm and self.tmm_location == "pre_bilstm":
            if m_raw is not None:
                m_aligned = self._align_motion_to_feat(m_raw, lgt, temp)
            else:
                # fallback: zero motion
                m_aligned = torch.zeros_like(z_tcn)
            z_clean, g = self.tmm(z_tcn, m_aligned) # (T', B, H) , (T', B, 1)
            z_for_lstm = z_clean
        else:
            z_for_lstm = z_tcn

        # ------------ BiLSTM ------------
        # tm_outputs = self.temporal_model(x, lgt)
        tm_outputs = self.temporal_model(z_for_lstm, lgt)
        z_seq = tm_outputs['predictions']           # (T’, B, H)


        ############################################
        # ---- TMM location: post_bilstm ----
        if self.enable_tmm and self.tmm_location == 'post_bilstm':
            if m_raw is not None:
                m_aligned = self._align_motion_to_feat(m_raw, lgt, temp)
            else:
                m_aligned = torch.zeros_like(z_seq)
            z_clean, g = self.tmm(z_seq, m_aligned)
            z_out = z_clean
        else:
            z_out = z_seq

        # ---- Classifiers ----
        seq_logits = self.classifier(z_out)                     # (T’, B, C)
        conv_logits = conv1d_outputs['conv_logits']             # (T’, B, C)

        # outputs = self.classifier(tm_outputs['predictions'])
        # outputs = self.classifier(tm_outputs['predictions'])


        pred = None if self.training \
            else self.decoder.decode(seq_logits, lgt, batch_first=False, probs=False)
        conv_pred = None if self.training \
            else self.decoder.decode(conv_logits, lgt, batch_first=False, probs=False)
        

        return {
            "framewise_features": framewise,
            "visual_features": z_tcn,
            "temproal_features": z_seq,
            "feat_len": lgt,
            "conv_logits": conv_logits,
            "sequence_logits": seq_logits,
            "conv_sents": conv_pred,
            "recognized_sents": pred,
            # OPTIONAL: expose gate for diagnostics
            "tmm_gate": g if (self.enable_tmm) else None,            
        }

    def criterion_calculation(self, ret_dict, label, label_lgt):
        loss = 0
        for k, weight in self.loss_weights.items():
            if k == 'ConvCTC':
                loss += weight * self.loss['CTCLoss'](ret_dict["conv_logits"].log_softmax(-1),
                                                      label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                                                      label_lgt.cpu().int()).mean()
            elif k == 'SeqCTC':
                loss += weight * self.loss['CTCLoss'](ret_dict["sequence_logits"].log_softmax(-1),
                                                      label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                                                      label_lgt.cpu().int()).mean()
            elif k == 'Dist':
                loss += weight * self.loss['distillation'](ret_dict["conv_logits"],
                                                           ret_dict["sequence_logits"].detach(),
                                                           use_blank=False)
        return loss

    def criterion_init(self):
        self.loss['CTCLoss'] = torch.nn.CTCLoss(reduction='none', zero_infinity=False)
        self.loss['distillation'] = SeqKD(T=8)
        return self.loss
