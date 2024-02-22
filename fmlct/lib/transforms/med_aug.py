import monai
monai.data.set_track_meta(False)
from timm.layers.helpers import to_3tuple
from .intensity import NanToZero, RandHistogramShift
from .from_fmcib import RandomResizedCrop3D

__all__ = ["CTAugmentation", "CTValAugmentation", "MRAugmentation", "MRValAugmentation"]


class CTAugmentation(monai.transforms.Compose):
    def __init__(self, size, prob=0.5, channel_dim='no_channel'):
        size = to_3tuple(size)
        super().__init__([
            monai.transforms.EnsureChannelFirst(channel_dim=channel_dim),
            # monai.transforms.RandGaussianSmooth(prob=prob),
            # monai.transforms.RandAffine(prob=prob, translate_range=[10, 10, 10]),
            monai.transforms.RandAxisFlip(prob=prob),
            monai.transforms.RandRotate90(prob=prob),
            NanToZero(),
            monai.transforms.ToTensor(track_meta=False),
        ])

class CTValAugmentation(monai.transforms.Compose):
    def __init__(self, size, channel_dim='no_channel'):
        size = to_3tuple(size)
        super().__init__([
            monai.transforms.EnsureChannelFirst(channel_dim=channel_dim),
            NanToZero(),
            monai.transforms.ToTensor(track_meta=False),
        ])

class MRAugmentation(monai.transforms.Compose):
    def __init__(self, size, prob=0.5, channel_dim='no_channel'):
        size = to_3tuple(size)
        super().__init__([
            monai.transforms.EnsureChannelFirst(channel_dim=channel_dim),
            # RandomResizedCrop3D(size=size[0]), # from_fmcib.py
            monai.transforms.RandAxisFlip(prob=prob),
            RandHistogramShift(prob=prob), # intensity.py
            monai.transforms.RandGaussianSmooth(prob=prob),
            monai.transforms.SpatialPad(spatial_size=size),
            NanToZero(),
            monai.transforms.ToTensor(track_meta=False),
        ])

class MRValAugmentation(monai.transforms.Compose):
    def __init__(self, size, channel_dim='no_channel'):
        size = to_3tuple(size)
        super().__init__([
            monai.transforms.EnsureChannelFirst(channel_dim=channel_dim),
            NanToZero(),
            monai.transforms.ToTensor(track_meta=False),
        ])