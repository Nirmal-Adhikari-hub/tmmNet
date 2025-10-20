import torch
import torch.nn as nn
import torch.nn.functional as F


class MotionDiffEncoder(nn.Module):
    """
    Tiny CNN over per-frame differences Δx_t (RGB), producing m̃_t ∈ R^d.
    Expects inputs normalized per-frame before differencing (eq. 4-5)
    """

    def __init__(self, in_ch=3, feat_dim=512):
        super().__init__()
        # lightweight convs; keep cheap
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1)), # pool spatial dimensions to 1x1
        )

        self.proj = nn.Linear(32, feat_dim)


    def forward(self, x): # x: (B, T, 3, H, W), uint8 or float in [0,1]
        B, T, C, H, W = x.shape
        # per-frame channel-wise standardization
        # (add small eps to avoid div by zero)
        eps = 1e-5
        x = x.float()
        mean = x.mean(dim=(2,3,4), keepdim=True)
        std = x.std(dim=(2,3,4), keepdim=True) + eps
        x_n = (x - mean) / std

        # Δx_t = norm(x_t) - norm(x_{t-1}); Δx_t = 0 (eq. 4) 
        diff = torch.zeros_like(x_n)
        diff[:,1:] = x_n[:,1:] - x_n[:,:-1]

        # run tiny CNN per frame
        diff = diff.view(B*T, C, H, W)  # (B*T, 3, H, W)
        feat = self.net(diff).view(B*T, -1)           # (B*T, 32)
        m = self.proj(feat).view(B, T, -1) # (B, T, d) (eq. 5)
        return m
    

class TemporalMotionMix(nn.Module):
    """
    TMM gate per timestep using [z_t || m̃_t], and two light projections (s_t, r_t),
    then mix per eq 6-8
    """
    def __init__(self, d, alpha=0.2):
        super().__init__()
        self.g_fc = nn.Linear(2*d, 1) # gate from concat [z || m̃]
        self.s_fc = nn.Linear(d, d) # stroke projection
        self.r_fc = nn.Linear(d, d) # transition projection
        self.alpha = alpha


    def forward(self, z, m):
        """
        z: (B, T, d) context features from TCN or BiLSTM
        m: (B, T, d) motion embeddings aligned to z
        returns:
            z_clean: (T, B, d)
        """
        T, B, d = z.shape
        cat = torch.cat([z, m], dim=-1) # (T, B, 2d)
        g = torch.sigmoid(self.g_fc(cat)).view(T, B, 1) # (T, B, 1) gate eq 6

        s = F.relu(self.s_fc(z))  # (T, B, d) eq 7
        r = F.relu(self.r_fc(z))  # (T, B, d) eq 7

        z_clean = (1 - g)*s + g*r + self.alpha*(1 - g)*z   # (Eq. 8)
        return z_clean, g