import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------
# Lightweight 3D cue extractor
# ---------------------------

class _GN(nn.GroupNorm):
    """GroupNorm with 1 group works like Layer/Instance Norm for conv features."""
    def __init__(self, num_channels: int):
        super().__init__(num_groups=1, num_channels=num_channels)


class _Basic3DBlock(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.conv1 = nn.Conv3d(c, c, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn1   = _GN(c)
        self.conv2 = nn.Conv3d(c, c, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn2   = _GN(c)

    def forward(self, x):
        out = F.relu(self.gn1(self.conv1(x)), inplace=True)
        out = self.gn2(self.conv2(out))
        return F.relu(out + x, inplace=True)


class MotionCue3D(nn.Module):
    """
    Tiny inflated 3D stem:
      - spatial downsample once, keep temporal stride = 1
      - a couple of residual blocks
      - global spatial pool -> T steps preserved
      - linear projection to 'feat_dim'
    Input:  x (B, T, 3, H, W)
    Output: m (B, T, d)
    """
    def __init__(self, in_ch=3, base_channels=32, num_blocks=2, feat_dim=512):
        super().__init__()
        C = base_channels
        # first layer: spatial stride 2, keep T
        self.conv0 = nn.Conv3d(in_ch, C, kernel_size=(3,3,3), stride=(1,2,2), padding=1, bias=False)
        self.gn0   = _GN(C)
        self.blocks = nn.Sequential(*[_Basic3DBlock(C) for _ in range(max(0, num_blocks))])
        self.proj  = nn.Linear(C, feat_dim)
        self._reset()

    def _reset(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if isinstance(m, (nn.GroupNorm, nn.BatchNorm3d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):  # (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous()    # -> (B, C, T, H, W)
        y = F.relu(self.gn0(self.conv0(x)), inplace=True)  # (B, Cb, T, H/2, W/2)
        y = self.blocks(y)                                  # (B, Cb, T, H/2, W/2)
        # global spatial pool (keep T)
        y = F.adaptive_avg_pool3d(y, output_size=(T, 1, 1)).squeeze(-1).squeeze(-1)  # (B, Cb, T)
        y = y.permute(0, 2, 1).contiguous()  # (B, T, Cb)
        m = self.proj(y)                     # (B, T, d)
        return m


# --------------------------------
# Temporal Motion Mix (vector gate)
# --------------------------------

class TemporalMotionMix(nn.Module):
    """
    Vector gate g_t ∈ R^d on (T, B, d) features with per-branch projections.
    z_clean = (1 - g)*s(z) + g*r(z) + alpha*(1 - g)*z
    """
    def __init__(self, d, alpha=1.0):
        super().__init__()
        self.ln_z = nn.LayerNorm(d)
        self.ln_m = nn.LayerNorm(d)
        self.g_fc = nn.Linear(2 * d, d)  # vector gate
        self.s_fc = nn.Linear(d, d)
        self.r_fc = nn.Linear(d, d)
        self.alpha = float(alpha)
        # neutral start for the gate
        nn.init.constant_(self.g_fc.bias, 0.0)

    @torch.no_grad()
    def set_alpha(self, alpha: float):
        self.alpha = float(alpha)

    def forward(self, z, m):
        """
        z, m: (T, B, d)
        Returns:
          z_clean: (T, B, d)
          g:       (T, B, d)  (post-sigmoid)
          g_pre:   (T, B, d)  (pre-sigmoid)
        """
        T, B, d = z.shape
        z = self.ln_z(z)
        m = self.ln_m(m)
        cat = torch.cat([z, m], dim=-1)         # (T, B, 2d)
        g_pre = self.g_fc(cat)                  # (T, B, d)
        g = torch.sigmoid(g_pre)                # (T, B, d)

        s = F.relu(self.s_fc(z), inplace=True)  # (T, B, d)
        r = F.relu(self.r_fc(z), inplace=True)  # (T, B, d)

        z_clean = (1.0 - g) * s + g * r + self.alpha * (1.0 - g) * z
        return z_clean, g, g_pre



# class MotionDiffEncoder(nn.Module):
#     """
#     Tiny CNN over per-frame differences Δx_t (RGB), producing m̃_t ∈ R^d.
#     Expects inputs normalized per-frame before differencing (eq. 4-5)
#     """

#     def __init__(self, in_ch=3, feat_dim=512):
#         super().__init__()
#         # lightweight convs; keep cheap
#         self.net = nn.Sequential(
#             nn.Conv2d(in_ch, 16, 3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(16, 32, 3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.AdaptiveAvgPool2d((1,1)), # pool spatial dimensions to 1x1
#         )

#         self.proj = nn.Linear(32, feat_dim)


#     def forward(self, x): # x: (B, T, 3, H, W), uint8 or float in [0,1]
#         B, T, C, H, W = x.shape
#         # per-frame channel-wise standardization
#         # (add small eps to avoid div by zero)
#         eps = 1e-5
#         x = x.float()
#         mean = x.mean(dim=(2,3,4), keepdim=True)
#         std = x.std(dim=(2,3,4), keepdim=True) + eps
#         x_n = (x - mean) / std

#         # Δx_t = norm(x_t) - norm(x_{t-1}); Δx_t = 0 (eq. 4) 
#         diff = torch.zeros_like(x_n)
#         diff[:,1:] = x_n[:,1:] - x_n[:,:-1]

#         # run tiny CNN per frame
#         diff = diff.view(B*T, C, H, W)  # (B*T, 3, H, W)
#         feat = self.net(diff).view(B*T, -1)           # (B*T, 32)
#         m = self.proj(feat).view(B, T, -1) # (B, T, d) (eq. 5)
#         return m
    

# class TemporalMotionMix(nn.Module):
#     """
#     TMM gate per timestep using [z_t || m̃_t], and two light projections (s_t, r_t),
#     then mix per eq 6-8
#     """
#     def __init__(self, d, alpha=0.2):
#         super().__init__()
#         self.g_fc = nn.Linear(2*d, 1) # gate from concat [z || m̃]
#         self.s_fc = nn.Linear(d, d) # stroke projection
#         self.r_fc = nn.Linear(d, d) # transition projection
#         self.alpha = alpha


#     def forward(self, z, m):
#         """
#         z: (B, T, d) context features from TCN or BiLSTM
#         m: (B, T, d) motion embeddings aligned to z
#         returns:
#             z_clean: (T, B, d)
#         """
#         T, B, d = z.shape
#         cat = torch.cat([z, m], dim=-1) # (T, B, 2d)
#         g = torch.sigmoid(self.g_fc(cat)).view(T, B, 1) # (T, B, 1) gate eq 6

#         s = F.relu(self.s_fc(z))  # (T, B, d) eq 7
#         r = F.relu(self.r_fc(z))  # (T, B, d) eq 7

#         z_clean = (1 - g)*s + g*r + self.alpha*(1 - g)*z   # (Eq. 8)
#         return z_clean, g


# class TemporalMotionMix(nn.Module):
#     def __init__(self, d, alpha=0.2):
#         super().__init__()
#         self.ln_z = nn.LayerNorm(d)
#         self.ln_m = nn.LayerNorm(d)
#         self.g_fc = nn.Linear(2*d, 1)
#         self.s_fc = nn.Linear(d, d)
#         self.r_fc = nn.Linear(d, d)
#         self.alpha = alpha
#         nn.init.constant_(self.g_fc.bias, 0.0)  # keep neutral start

#     def forward(self, z, m):
#         # z, m: (T, B, d)
#         z = self.ln_z(z)
#         m = self.ln_m(m)
#         cat = torch.cat([z, m], dim=-1)
#         g_pre = self.g_fc(cat)
#         g = torch.sigmoid(g_pre).view(z.size(0), z.size(1), 1)
#         s = F.relu(self.s_fc(z))
#         r = F.relu(self.r_fc(z))
#         z_clean = (1 - g)*s + g*r + self.alpha*(1 - g)*z
#         return z_clean, g, g_pre
