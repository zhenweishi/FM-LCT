import numpy as np
import SimpleITK as sitk
from pathlib import Path

import torch

__all__ = [
    "read_image",
    "read_array",
    "write_image",
    "write_array",
]

def read_image(path: str) -> sitk.Image:
    return sitk.ReadImage(str(path))
    
def read_array(path: str) -> np.ndarray:
    return sitk.GetArrayFromImage(read_image(path))

def write_image(path: str, img: sitk.Image) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(path))

def write_array(path: str, arr: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if isinstance(arr, sitk.Image):
        write_image(path, arr)
        return 
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    write_image(path, sitk.GetImageFromArray(arr))