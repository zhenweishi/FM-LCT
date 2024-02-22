"""
This source code is based on vits.py, which is licensed under the Attribution-NonCommercial 4.0 International License.
Original source code: https://github.com/facebookresearch/moco-v3/blob/main/vits.py
"""


import math
import torch
import torch.nn as nn
from functools import partial, reduce
from operator import mul
# timm==0.9.10
from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.layers import to_3tuple
from timm.models.layers import PatchEmbed

__all__ = [
    'vit_3d_small', 
    'vit_3d_base',
    'vit_3d_conv_small',
    'vit_3d_conv_base',
]


class VisionTransformerMoCo3D(VisionTransformer):
    def __init__(self, stop_grad_conv1=False, **kwargs):
        super().__init__(**kwargs)
        # Use fixed 3D sin-cos position embedding
        self.build_3d_sincos_position_embedding()

        # weight initialization
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if 'qkv' in name:
                    # treat the weights of Q, K, V separately
                    val = math.sqrt(6. / float(m.weight.shape[0] // 3 + m.weight.shape[1]))
                    nn.init.uniform_(m.weight, -val, val)
                else:
                    nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.cls_token, std=1e-6)

        if isinstance(self.patch_embed, PatchEmbed):
            # xavier_uniform initialization
            val = math.sqrt(6. / float(3 * reduce(mul, self.patch_embed.patch_size, 1) + self.embed_dim))
            nn.init.uniform_(self.patch_embed.proj.weight, -val, val)
            nn.init.zeros_(self.patch_embed.proj.bias)

            if stop_grad_conv1:
                self.patch_embed.proj.weight.requires_grad = False
                self.patch_embed.proj.bias.requires_grad = False

    def build_3d_sincos_position_embedding(self, temperature=10000.):
        d, h, w = self.patch_embed.grid_size
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_d = torch.arange(d, dtype=torch.float32)
        grid_w, grid_h, grid_d = torch.meshgrid(grid_w, grid_h, grid_d)
        assert self.embed_dim % 6 == 0, 'Embed dimension must be divisible by 6 for 3D sin-cos position embedding'
        pos_dim = self.embed_dim // 6
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature**omega)
        out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
        out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
        out_d = torch.einsum('m,d->md', [grid_d.flatten(), omega])
        pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h), torch.sin(out_d), torch.cos(out_d)], dim=1)[None, :, :]

        assert self.num_prefix_tokens == 1, 'Assuming one and only one token, [cls]'
        pe_token = torch.zeros([1, 1, self.embed_dim], dtype=torch.float32)
        self.pos_embed = nn.Parameter(torch.cat([pe_token, pos_emb], dim=1))
        self.pos_embed.requires_grad = False


class ConvStem3D(nn.Module):
    """ 
    ConvStem, from Early Convolutions Help Transformers See Better, Tete et al. https://arxiv.org/abs/2106.14881
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True, **kwargs):
        super().__init__()

        # assert patch_size == 16, 'ConvStem only supports patch size of 16'
        # assert embed_dim % 8 == 0, 'Embed dimension must be divisible by 8 for ConvStem'

        img_size = to_3tuple(img_size)
        patch_size = to_3tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2] // patch_size[2])
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        self.flatten = flatten

        # build stem, similar to the design in https://arxiv.org/abs/2106.14881
        assert patch_size[0] & (patch_size[0] - 1) == 0, 'Patch size must be a power of 2.'
        n_conv = int(math.log2(patch_size[0]))

        stem = []
        input_dim, output_dim = in_chans, embed_dim // 2 ** (n_conv - 1)
        for l in range(n_conv):
            stem.append(nn.Conv3d(input_dim, output_dim, kernel_size=3, stride=2, padding=1, bias=False))
            stem.append(nn.BatchNorm3d(output_dim))
            stem.append(nn.ReLU(inplace=True))
            input_dim = output_dim
            output_dim *= 2
        stem.append(nn.Conv3d(input_dim, embed_dim, kernel_size=1))
        self.proj = nn.Sequential(*stem)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, D, H, W = x.shape
        assert D == self.img_size[0] and H == self.img_size[1] and W == self.img_size[2], \
            f"Input image size ({D}*{H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]}*{self.img_size[2]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHWD -> BNC
        x = self.norm(x)
        return x


def vit_3d_small(**kwargs):
    model = VisionTransformerMoCo3D(
        patch_size=16, embed_dim=384, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

def vit_3d_base(**kwargs):
    model = VisionTransformerMoCo3D(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

def vit_3d_conv_small(**kwargs):
    # minus one ViT block
    model = VisionTransformerMoCo3D(
        patch_size=16, embed_dim=384, depth=11, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), embed_layer=ConvStem3D, **kwargs)
    model.default_cfg = _cfg()
    return model

def vit_3d_conv_base(**kwargs):
    # minus one ViT block
    model = VisionTransformerMoCo3D(
        patch_size=16, embed_dim=768, depth=11, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), embed_layer=ConvStem3D, **kwargs)
    model.default_cfg = _cfg()
    return model