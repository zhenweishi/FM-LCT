import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import random
import warnings
from functools import partial as _partial

__all__ = [
    "partial",
    "set_seed",
]

def partial(func, *args, **kwargs):
    return _partial(func, *args, **kwargs)

def set_seed(seed=None):
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        # np.random.seed(seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    else:
        cudnn.benchmark = True
