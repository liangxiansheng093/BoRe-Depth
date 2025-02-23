import torch
import torch.nn as nn
from .MPViT_encoder import MPViT, _cfg_mpvit


class Seg_Encoder(nn.Module):

    def __init__(self):
        super(Seg_Encoder, self).__init__()
        self.backbone = MPViT(
            num_stages=4,
            num_path=[2, 3, 3, 3],
            num_layers=[1, 2, 4, 1],
            embed_dims=[64, 96, 176, 216],
            mlp_ratios=[4, 4, 4, 4],
            num_heads=[8, 8, 8, 8],
        )
        self.backbone.default_cfg = _cfg_mpvit()

    def init_weights(self):
        pass

    def forward(self, x):

        features = self.encoder(x)

        return features
