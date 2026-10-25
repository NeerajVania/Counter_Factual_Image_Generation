import torch
import torch.nn as nn
import math

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps):
        device = timesteps.device
        half_dim = self.embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)
        return emb 


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim_q, dim_kv, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim_q, kdim=dim_kv, vdim=dim_kv, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim_q)

    def forward(self, x, context):
        attn_output, _ = self.attn(x, context, context)
        return self.norm(x + attn_output)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim=None):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        if time_emb_dim is not None:
            self.time_mlp = nn.Sequential(
                nn.Linear(time_emb_dim, out_channels),
                nn.ReLU()
            )

    def forward(self, x, t_emb=None):
        h = self.conv(x)
        if t_emb is not None:
            t_emb = self.time_mlp(t_emb)
            h = h + t_emb[:, :, None, None]
        return h

class EmbedFC(nn.Module):
  def __init__(self, in_dim: int, emb_dim: int) -> None:
    super(EmbedFC, self).__init__()
    self.in_dim = in_dim
    self.emb_dim = emb_dim
    layers = [
      nn.Linear(in_dim, emb_dim),
      nn.GELU(),
      nn.Linear(emb_dim, emb_dim)
    ]
    self.model = nn.Sequential(*layers)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = x.view(-1, self.in_dim)
    return self.model(x)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, base_channels, label_emb_dim, num_classes, time_emb_dim):
        super().__init__()
        self.label_emb = EmbedFC(num_classes, label_emb_dim)

        self.time_emb = SinusoidalPositionEmbeddings(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )

        self.enc1 = ConvBlock(in_channels, base_channels, time_emb_dim)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(base_channels, base_channels * 2, time_emb_dim)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4, time_emb_dim)
        self.pool3 = nn.MaxPool2d(2)
        

        self.bottleneck = ConvBlock(base_channels * 4, base_channels * 8, time_emb_dim)
        self.cross_attn = CrossAttentionBlock(
            dim_q=base_channels * 8,
            dim_kv=label_emb_dim,
            num_heads=8
        )

        self.up1 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_channels * 8, base_channels * 4, time_emb_dim)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_channels * 4, base_channels * 2, time_emb_dim)
        self.up3 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(base_channels * 2, base_channels, time_emb_dim)

        self.final = nn.Conv2d(base_channels, out_channels, kernel_size=1)



    def forward(self, x, labels, timesteps, context_mask):
        t_emb = self.time_emb(timesteps)
        t_emb = self.time_mlp(t_emb)

        e1 = self.enc1(x, t_emb)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1, t_emb)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2, t_emb)
        p3 = self.pool3(e3)

        b = self.bottleneck(p3, t_emb)

        B, C, H, W = b.shape
        b_flat = b.view(B, C, -1).permute(0, 2, 1)

        label_embed = self.label_emb(labels.float())

        if context_mask is not None:
            context_mask = context_mask.to(torch.float32).view(-1, 1)
            label_embed = label_embed * (1 - context_mask)
            # Make sure not all labels are masked:
            if context_mask.sum() == context_mask.numel():
                context_mask[0] = 0
                label_embed[0] = self.label_emb(labels[0:1].float())
        label_embed = label_embed.unsqueeze(1)

        b_attn = self.cross_attn(b_flat, label_embed)

        b = b_attn.permute(0, 2, 1).view(B, C, H, W)
        u1 = self.up1(b)
        d1 = self.dec1(torch.cat([u1, e3], dim=1), t_emb)

        u2 = self.up2(d1)
        d2 = self.dec2(torch.cat([u2, e2], dim=1), t_emb)
        u3 = self.up3(d2)
        d3 = self.dec3(torch.cat([u3, e1], dim=1), t_emb)
        out = self.final(d3)

        return out