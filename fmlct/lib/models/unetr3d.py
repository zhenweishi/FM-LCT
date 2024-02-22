from matplotlib.pyplot import grid
from requests import patch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import numpy as np

from timm.layers.helpers import to_3tuple
from fmlct.lib.layers.position_embed import build_3d_sincos_position_embedding
from fmlct.lib.layers.patch_embed import PatchEmbed3D

import time

__all__ = ["UNETR3D"]


class UNETR3D(nn.Module):
    """General segmenter module for 3D medical images
    """
    def __init__(self, encoder, decoder, args):
        super().__init__()
        if args.spatial_dim == 3:
            input_size = (args.roi_x, args.roi_y, args.roi_z)
        elif args.spatial_dim == 2:
            input_size = (args.roi_x, args.roi_y)

        self.encoder = encoder(img_size=input_size,
                               patch_size=args.patch_size,
                               in_chans=args.in_chans,
                               embed_dim=args.encoder_embed_dim,
                               depth=args.encoder_depth,
                               num_heads=args.encoder_num_heads,
                               drop_path_rate=args.drop_path,
                               embed_layer=PatchEmbed3D,
                               )
        self.decoder = decoder(in_channels=args.in_chans,
                               out_channels=args.num_classes,
                               img_size=input_size,
                               patch_size=args.patch_size,
                               feature_size=args.feature_size,
                               hidden_size=args.encoder_embed_dim,
                               spatial_dims=args.spatial_dim)
    
    def get_num_layers(self):
        return len(self.encoder.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        total_set = set()
        module_prefix_dict = {self.encoder: 'encoder',
                              self.decoder: 'decoder'}
        for module, prefix in module_prefix_dict.items():
            if hasattr(module, 'no_weight_decay'):
                for name in module.no_weight_decay():
                    total_set.add(f'{prefix}.{name}')
        print(f"{total_set} will skip weight decay")
        return total_set
    
    def forward(self, x_in, time_meters=None):
        """
        x_in in shape of [BCHWD]
        """
        s_time = time.perf_counter()
        # x, hidden_states = self.encoder(x_in, time_meters=time_meters)
        hidden_states = self.encoder.get_intermediate_layers(x_in, n=9999) # n>=len(blocks) means return all layers
        x = self.encoder.norm(hidden_states[-1])
        
        if time_meters is not None:
            torch.cuda.synchronize()
            time_meters['enc'].append(time.perf_counter() - s_time)

        s_time = time.perf_counter()
        logits = self.decoder(x_in, x, hidden_states)
        if time_meters is not None:
            torch.cuda.synchronize()
            time_meters['dec'].append(time.perf_counter() - s_time)
        return logits
