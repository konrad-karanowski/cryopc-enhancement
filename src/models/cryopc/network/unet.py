from typing import Optional

import torch
from torch import nn, einsum
from abc import abstractmethod
from einops import rearrange, repeat
from functools import partial, wraps
import einops
import numpy as np

from packaging import version
from collections import namedtuple

import torch.nn.functional as F
from src.models.cryopc.network.blocks import ResidualBlock, LinearAttention, TransformerCrossAttentionBlock, StructEmbedSequential, apply_norm

class Upsample(nn.Module):
    def __init__(self, channels, use_conv):
        super(Upsample, self).__init__()
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv3d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


# downsample
class Downsample(nn.Module):
    def __init__(self, channels, use_conv):
        super(Downsample, self).__init__()
        self.use_conv = use_conv
        if use_conv:
            self.op = nn.Conv3d(channels, channels, kernel_size=3, stride=2, padding=1)
        else:
            self.op = nn.AvgPool3d(stride=2, kernel_size=2)

    def forward(self, x):
        return self.op(x)

# The full UNet model with attention
class UNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        model_channels=32,
        out_channels=1,
        num_res_blocks=2,
        attention_resolutions=(1, 2, 4),
        dropout=0.2,
        channel_mult=(1, 2, 2, 4),        
        conv_resample=True,
        num_heads=4,
        struc_dim=512,
        feature_extractor: nn.Module = None,
        *args,
        **kwargs
    ) -> None:
        super(UNet, self).__init__()

        # For conditioning on point clouds, coordinates and other shenanigans
        self.feature_extractor = feature_extractor

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_heads = num_heads
        self.struc_dim = struc_dim

        # down blocks
        self.down_blocks = nn.ModuleList([
            StructEmbedSequential(nn.Conv3d(in_channels, model_channels, kernel_size=3, padding=1))
        ])
        down_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResidualBlock(ch, mult * model_channels, dropout)
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(LinearAttention(ch, num_heads=num_heads))
                    # layers.append(TransformerCrossAttentionBlock(ch, struc_dim, num_heads=num_heads))
                self.down_blocks.append(StructEmbedSequential(*layers))
                down_block_chans.append(ch)
            if level != len(channel_mult) - 1:  # don't use downsample for the last layer
                self.down_blocks.append(StructEmbedSequential(Downsample(ch, conv_resample)))
                down_block_chans.append(ch)
                ds *= 2

        # middle block
        self.middle_block = StructEmbedSequential(
            ResidualBlock(ch, ch, dropout),
            LinearAttention(ch, num_heads=num_heads),
            TransformerCrossAttentionBlock(ch, struc_dim, num_heads=num_heads),
            ResidualBlock(ch, ch, dropout)
        )

        # up blocks
        self.up_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [
                    ResidualBlock(
                        ch + down_block_chans.pop(),
                        model_channels * mult,
                        dropout
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(LinearAttention(ch, num_heads=num_heads))
                    # layers.append(TransformerCrossAttentionBlock(ch, struc_dim, num_heads=num_heads))
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch, conv_resample))
                    ds //= 2
                self.up_blocks.append(StructEmbedSequential(*layers))

        self.out = nn.Sequential(
            apply_norm(ch),
            nn.SiLU(),
            nn.Conv3d(model_channels, out_channels, kernel_size=3, padding=1),
        )

    def _extract_struc_feat(self, struc_feat: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if struc_feat is not None and self.feature_extractor is not None:
            struc_feat, _, _ = self.feature_extractor(struc_feat)
            struc_feat = struc_feat.unsqueeze(1) # make it [bs, 1, emb_dim]
        else:
            struc_feat = None
        return struc_feat


    def forward(self, x, struc_feat=None, *args, **kwargs):
        """
        Apply the model to an input batch.
        x: [N x C x D x H x W] Tensor of inputs.
        struc_feat: [B, target_len, emb_dim] structure embeddings
        """ 
        hs = []
        
        # down stage
        h = x
        for module in self.down_blocks:
            h = module(h)
            hs.append(h)
        
        # Extract features
        context = self._extract_struc_feat(struc_feat)

        h = self.middle_block(h, struc_feat=context)
        
        # up stage
        for module in self.up_blocks:
            cat_in = torch.cat([h, hs.pop()], dim=1)
            h = module(cat_in)   
                  
        return self.out(h)


if __name__ == '__main__':
    from src.models.cryopc.network.pointnet import PointNetfeat

    model = UNet(
        in_channels=1,
        model_channels=32,
        out_channels=1,
        num_res_blocks=2,
        attention_resolutions=[1, 2, 4],
        dropout=0.2,
        channel_mult=[1, 2, 2, 4],     
        conv_resample=True,
        num_heads=4,
        struc_dim=512,
        feature_extractor=None,
    )

    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
