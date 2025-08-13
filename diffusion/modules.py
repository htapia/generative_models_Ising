import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.img_size = img_size
        self.device = device

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n, labels, cfg_scale=3):
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 1, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, labels)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = x.clamp(-1, 1)
        return x
  

import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        # Reshape: (batch, height*width, channels)
        x_flat = x.view(batch_size, channels, height * width).transpose(1, 2)
        
        # Attention
        x_ln = self.ln(x_flat)
        attn_out, _ = self.mha(x_ln, x_ln, x_ln)
        attn_out = attn_out + x_flat
        
        # Feed forward
        ff_out = self.ff_self(attn_out) + attn_out
        
        # Reshape back: (batch, channels, height, width)
        return ff_out.transpose(1, 2).view(batch_size, channels, height, width)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels),
        )

    def forward(self, x, emb, y_emb=None):
        x = self.maxpool_conv(x)
        # Procesar embedding combinado
        if y_emb is not None:
            combined_emb = emb + y_emb  # Suma de embeddings
        else:
            combined_emb = emb
            
        emb_out = self.emb_layer(combined_emb)
        emb_out = emb_out.unsqueeze(-1).unsqueeze(-1)
        emb_out = emb_out.expand(-1, -1, x.shape[-2], x.shape[-1])
        return x + emb_out


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels),
        )

    def forward(self, x, skip_x, emb, y_emb=None):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        
        # Procesar embedding combinado
        if y_emb is not None:
            combined_emb = emb + y_emb
        else:
            combined_emb = emb
            
        emb_out = self.emb_layer(combined_emb)
        emb_out = emb_out.unsqueeze(-1).unsqueeze(-1)
        emb_out = emb_out.expand(-1, -1, x.shape[-2], x.shape[-1])
        return x + emb_out


class UNet_conditional(nn.Module):
    def __init__(self, c_in=1, c_out=1, time_dim=256, num_classes=None, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128, 32)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256, 16)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256, 8)

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128, 16)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64, 32)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64, 64)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)
        
        # Procesamiento de embeddings
        self.label_emb = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

    def pos_encoding(self, t, channels):
        # t shape: (batch_size, 1)
        batch_size = t.shape[0]
        
        # Crear frecuencias
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2, device=self.device).float() / channels))
        
        # Broadcasting correcto
        t_expanded = t * inv_freq.unsqueeze(0)  # (batch_size, channels//2)
        
        # Calcular sin/cos
        pos_enc_sin = torch.sin(t_expanded)
        pos_enc_cos = torch.cos(t_expanded)
        
        # Intercalar sin/cos para obtener dimensión completa
        pos_enc = torch.zeros(batch_size, channels, device=self.device)
        pos_enc[:, 0::2] = pos_enc_sin
        pos_enc[:, 1::2] = pos_enc_cos
        
        return pos_enc

    def forward(self, x, t, y):
        # Procesar timestep
        t = t.unsqueeze(-1).float()
        t_emb = self.pos_encoding(t, self.time_dim)
        
        # Procesar labels
        y_emb = None
        if y is not None:
            y = y.float()
            if y.dim() == 1:
                y = y.unsqueeze(-1)
            y_emb = self.label_emb(y)

        # U-Net forward pass
        x1 = self.inc(x)
        x2 = self.down1(x1, t_emb, y_emb)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t_emb, y_emb)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t_emb, y_emb)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t_emb, y_emb)
        x = self.sa4(x)
        x = self.up2(x, x2, t_emb, y_emb)
        x = self.sa5(x)
        x = self.up3(x, x1, t_emb, y_emb)
        x = self.sa6(x)
        output = self.outc(x)
        return output