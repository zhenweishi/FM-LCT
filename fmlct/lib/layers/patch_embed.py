from abc import ABC
from typing import Callable, List, Optional, Tuple, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.layers.helpers import to_2tuple, to_3tuple
from timm.layers.trace_utils import _assert
from .deformable_conv_3d import DeformConv3d

__all__ = ['PatchEmbedABC', 'PatchEmbed', 'PatchEmbed3D', 'PatchEmbed3D_Deform', 'ConvStem', 'ConvStem3D']

class PatchEmbedABC(nn.Module, ABC):
    """ Abstract base class for Patch Embeddingq
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
    
class PatchEmbed(PatchEmbedABC):
    """ 2D Image to Patch Embedding
    """
    dynamic_img_pad: torch.jit.Final[bool]

    def __init__(
            self,
            img_size: Optional[int] = 224,
            patch_size: int = 16,
            in_chans: int = 3,
            embed_dim: int = 768,
            norm_layer: Optional[Callable] = None,
            flatten: bool = True,
            bias: bool = True,
            strict_img_size: bool = True,
            dynamic_img_pad: bool = False,
            proj=nn.Conv2d,
    ):
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        if img_size is not None:
            self.img_size = to_2tuple(img_size)
            self.grid_size = tuple([s // p for s, p in zip(self.img_size, self.patch_size)])
            self.num_patches = self.grid_size[0] * self.grid_size[1]
        else:
            self.img_size = None
            self.grid_size = None
            self.num_patches = None

        self.flatten = flatten
        self.strict_img_size = strict_img_size
        self.dynamic_img_pad = dynamic_img_pad

        self.proj = proj(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        if self.img_size is not None:
            if self.strict_img_size:
                _assert(H == self.img_size[0], f"Input height ({H}) doesn't match model ({self.img_size[0]}).")
                _assert(W == self.img_size[1], f"Input width ({W}) doesn't match model ({self.img_size[1]}).")
            elif not self.dynamic_img_pad:
                _assert(
                    H % self.patch_size[0] == 0,
                    f"Input height ({H}) should be divisible by patch size ({self.patch_size[0]})."
                )
                _assert(
                    W % self.patch_size[1] == 0,
                    f"Input width ({W}) should be divisible by patch size ({self.patch_size[1]})."
                )
        if self.dynamic_img_pad:
            pad_h = (self.patch_size[0] - H % self.patch_size[0]) % self.patch_size[0]
            pad_w = (self.patch_size[1] - W % self.patch_size[1]) % self.patch_size[1]
            x = F.pad(x, (0, pad_w, 0, pad_h))
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        x = self.norm(x)
        return x

class PatchEmbed3D(PatchEmbedABC):
    """ 3D Image to Patch Embedding """
    dynamic_img_pad: torch.jit.Final[bool]

    def __init__(
        self,
        img_size: Optional[int] = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[Callable] = None,
        flatten: bool = True,
        in_chan_last: bool = False,
        proj=nn.Conv3d,
        bias: bool = True,
        strict_img_size: bool = True,
        dynamic_img_pad: bool = False
    ):
        super().__init__()
        self.img_size = to_3tuple(img_size) if img_size is not None else None
        self.patch_size = to_3tuple(patch_size)
        self.grid_size = [s // p for s, p in zip(self.img_size, self.patch_size)] if self.img_size else None
        self.num_patches = np.prod(self.grid_size) if self.grid_size else None
        self.in_chans = in_chans
        self.flatten = flatten
        self.in_chan_last = in_chan_last
        self.strict_img_size = strict_img_size
        self.dynamic_img_pad = dynamic_img_pad

        self.proj = proj(in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W, D = x.shape
        if self.img_size is not None:
            if self.strict_img_size:
                _assert(H == self.img_size[0] and W == self.img_size[1] and D == self.img_size[2], 
                        f"Input size ({H}*{W}*{D}) doesn't match model ({self.img_size[0]}*{self.img_size[1]}*{self.img_size[2]}).")
        if self.dynamic_img_pad:
            pad_h = (self.patch_size[0] - H % self.patch_size[0]) % self.patch_size[0]
            pad_w = (self.patch_size[1] - W % self.patch_size[1]) % self.patch_size[1]
            pad_d = (self.patch_size[2] - D % self.patch_size[2]) % self.patch_size[2]
            x = F.pad(x, (0, pad_d, 0, pad_w, 0, pad_h))
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHWD -> BNC
        x = self.norm(x)
        return x
    
class PatchEmbed3D_Deform(PatchEmbed3D):
    """ 3D Image to Patch Embedding with deformable conv
    """
    def __init__(self, proj=DeformConv3d, **kwargs):
        super().__init__(proj=proj, **kwargs)

class ConvStem(nn.Module):
    """ 
    ConvStem, from Early Convolutions Help Transformers See Better, Tete et al. https://arxiv.org/abs/2106.14881
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True, **kwargs):
        super().__init__()

        assert patch_size == 16, 'ConvStem only supports patch size of 16'
        assert embed_dim % 8 == 0, 'Embed dimension must be divisible by 8 for ConvStem'

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        # build stem, similar to the design in https://arxiv.org/abs/2106.14881
        stem = []
        input_dim, output_dim = 3, embed_dim // 8
        for l in range(4):
            stem.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=2, padding=1, bias=False))
            stem.append(nn.BatchNorm2d(output_dim))
            stem.append(nn.ReLU(inplace=True))
            input_dim = output_dim
            output_dim *= 2
        stem.append(nn.Conv2d(input_dim, embed_dim, kernel_size=1))
        self.proj = nn.Sequential(*stem)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x
    
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


# class PatchEmbed2D(PatchEmbedABC):
#     """ 2D Image to Patch Embedding
#     """
#     def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True, in_chan_last=True, proj=nn.Conv2d, bias=True):
#         super().__init__()
#         img_size = to_2tuple(img_size)
#         patch_size = to_2tuple(patch_size)
#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.grid_size = []
#         for im_size, pa_size in zip(img_size, patch_size):
#             self.grid_size.append(im_size // pa_size)
#         # self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2] // patch_size[2])
#         self.in_chans = in_chans
#         self.num_patches = np.prod(self.grid_size)
#         self.flatten = flatten
#         self.in_chan_last = in_chan_last

#         # self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
#         self.proj = proj(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
#         self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

#     def forward(self, x):
#         B, L, S = x.shape
#         assert S == np.prod(self.img_size) * self.in_chans, \
#             f"Input image total size {S} doesn't match model configuration"
#         if self.in_chan_last:
#             x = x.reshape(B * L, *self.img_size, self.in_chans).permute(0, 3, 1, 2) # When patchification follows HWC
#         else:
#             x = x.reshape(B * L, self.in_chans, *self.img_size)
#         x = self.proj(x)
#         if self.flatten:
#             x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
#         x = self.norm(x)
#         return x
#
# class PatchEmbed3D(PatchEmbedABC):
#     """ 3D Image to Patch Embedding
#     """
#     def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True, in_chan_last=False, proj=nn.Conv3d, bias=True):
#         super().__init__()
#         img_size = to_3tuple(img_size)
#         patch_size = to_3tuple(patch_size)
#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.grid_size = []
#         for im_size, pa_size in zip(img_size, patch_size):
#             self.grid_size.append(im_size // pa_size)
#         # self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2] // patch_size[2])
#         self.in_chans = in_chans
#         self.num_patches = np.prod(self.grid_size)
#         self.flatten = flatten
#         self.in_chan_last = in_chan_last

#         # self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
#         self.proj = proj(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
#         self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

#     def forward(self, x):
#         B, C, H, W, D = x.shape
#         assert H == self.img_size[0] and W == self.img_size[1] and D == self.img_size[2], \
#             f"Input image size ({H}*{W}*{D}) doesn't match model ({self.img_size[0]}*{self.img_size[1]}*{self.img_size[2]})."
#         x = self.proj(x)
#         if self.flatten:
#             x = x.flatten(2).transpose(1, 2)  # BCHWD -> BNC
#         x = self.norm(x)
#         return x