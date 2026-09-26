import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class SiLU(nn.Module):
    def forward(self, x):
        return F.silu(x, inplace=False)


def make_gn(num_channels: int, num_groups: int = 32) -> nn.GroupNorm:
    groups = min(num_groups, num_channels)
    while num_channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.in_ch, self.out_ch = in_ch, out_ch
        self.norm1 = make_gn(in_ch)
        self.act1 = SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.norm2 = make_gn(out_ch)
        self.act2 = SiLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        h = self.conv1(self.act1(self.norm1(x)))
        h = self.conv2(self.dropout(self.act2(self.norm2(h))))
        return h + self.skip(x)


class AttnBlock(nn.Module):
    """Single-head spatial self-attention (as in SD VAE)."""
    def __init__(self, channels: int):
        super().__init__()
        self.norm = make_gn(channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        h_ = self.norm(x)
        q = self.q(h_).reshape(b, c, h*w).permute(0, 2, 1)          # (b, hw, c)
        k = self.k(h_).reshape(b, c, h*w)                            # (b, c, hw)
        v = self.v(h_).reshape(b, c, h*w).permute(0, 2, 1)          # (b, hw, c)
        attn = torch.softmax(torch.bmm(q, k) / (c ** 0.5), dim=-1)  # (b, hw, hw)
        h_attn = torch.bmm(attn, v).permute(0, 2, 1).reshape(b, c, h, w)
        return x + self.proj(h_attn)


class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, stride=2, padding=1)
    def forward(self, x):
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self,
                 in_channels,
                 base_channels,     
                 ch_mult,
                 num_res_blocks,
                 latent_channels,
                 dropout,
                 use_attn_at
                 ):
        super().__init__()

        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        in_ch = base_channels
        blocks = []
        attn_idx = 0
        for i, mult in enumerate(ch_mult):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                blocks.append(ResBlock(in_ch, out_ch, dropout))
                in_ch = out_ch
            if i in use_attn_at:
                blocks.append(AttnBlock(in_ch))
            if i != len(ch_mult) - 1:
                blocks.append(Downsample(in_ch))
        self.down = nn.Sequential(*blocks)

        mid_ch = in_ch
        self.mid = nn.Sequential(
            ResBlock(mid_ch, mid_ch, dropout),
            AttnBlock(mid_ch),
            ResBlock(mid_ch, mid_ch, dropout),
        )

        self.norm_out = make_gn(mid_ch)
        self.act = SiLU()
        self.conv_mu = nn.Conv2d(mid_ch, latent_channels, 3, padding=1)
        self.conv_logvar = nn.Conv2d(mid_ch, latent_channels, 3, padding=1)

    def forward(self, x):
        h = self.conv_in(x)
        h = self.down(h)
        h = self.mid(h)
        h = self.act(self.norm_out(h))
        mu = self.conv_mu(h)        
        logvar = self.conv_logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self,
                 out_channels,
                 base_channels,
                 ch_mult,
                 num_res_blocks,
                 latent_channels,
                 dropout: float = 0.0,
                 use_attn_at: List[int] = [1, 2]
                 ):
        super().__init__()

        top_ch = base_channels * ch_mult[-1]
        self.conv_in = nn.Conv2d(latent_channels, top_ch, 3, padding=1)

        self.mid = nn.Sequential(
            ResBlock(top_ch, top_ch, dropout),
            AttnBlock(top_ch),
            ResBlock(top_ch, top_ch, dropout),
        )

        blocks = []
        in_ch = top_ch
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                blocks.append(ResBlock(in_ch, out_ch, dropout))
                in_ch = out_ch
            if i in use_attn_at:
                blocks.append(AttnBlock(in_ch))
            if i != 0:
                blocks.append(Upsample(in_ch))
        self.up = nn.Sequential(*blocks)

        self.norm_out = make_gn(in_ch)
        self.act = SiLU()
        self.conv_out = nn.Conv2d(in_ch, out_channels, 3, padding=1)

    def forward(self, z):
        h = self.conv_in(z)
        h = self.mid(h)
        h = self.up(h)
        h = self.act(self.norm_out(h))
        return self.conv_out(h)

class VAE(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 out_channels: int = 3,
                 base_channels: int = 128,
                 ch_mult: List[int] = [1, 2, 4, 8], 
                 num_res_blocks: int = 2,
                 latent_channels: int = 4,
                 dropout: float = 0.0,
                 use_attn_at_enc: List[int] = [3],
                 use_attn_at_dec: List[int] = [3],
                 ): 
        super().__init__()
        print("Initializing VAE Model with parameters:")
        print(f"  in_channels: {in_channels}")
        print(f"  out_channels: {out_channels}")
        print(f"  base_channels: {base_channels}")
        print(f"  ch_mult: {ch_mult}")
        print(f"  num_res_blocks: {num_res_blocks}")
        print(f"  latent_channels: {latent_channels}")
        print(f"  dropout: {dropout}")
        print(f"  use_attn_at_enc: {use_attn_at_enc}")
        print(f"  use_attn_at_dec: {use_attn_at_dec}")
        self.encoder = Encoder(in_channels, base_channels, ch_mult, num_res_blocks,
                               latent_channels, dropout, use_attn_at_enc)
        self.decoder = Decoder(out_channels, base_channels, ch_mult, num_res_blocks,
                               latent_channels, dropout, use_attn_at_dec)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar