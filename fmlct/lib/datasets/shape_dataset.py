import sys
sys.path.append('lib/')
sys.path.append('')
import fmlct.lib as lib
import fmlct.lib.datasets as datasets

import os
from pathlib import Path
from easydict import EasyDict as edict
from typing import Union, Tuple
import pandas as pd
import SimpleITK as sitk
import numpy as np
import torch
import scipy
import monai
monai.data.set_track_meta(False)

class ShapeDataset(datasets.PatchDataset):
    def __init__(self, shape_feat_col_start, shape_feat_col_end, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._shape_feat_col_start = shape_feat_col_start
        self._shape_feat_col_end = shape_feat_col_end
        # print(self._df.columns[shape_feat_col_start:shape_feat_col_end])

    def __getitem__(self, index):
        x, label = super().__getitem__(index)
        shape_feat = self._df.iloc[index, self._shape_feat_col_start:self._shape_feat_col_end].values.astype(np.float32)
        return (x, shape_feat), label

def test2():
    from pathlib import Path
    import shutil

    csv_path = "/media/wzt/plum14t/wzt/ProcressedData/imm/imm_pre_merged/ROI/all_roi_resize48/scaled_shape_features.csv"
    enable_negatives = False
    size = 48
    only_roi = False
    method = "resize"
    idx = 58
    shape_feat_col_start = 19
    shape_feat_col_end = shape_feat_col_start + 15

    output_dir = Path("/mnt/tmp/0")
    output_dir.mkdir(exist_ok=True, parents=True)  

    dataset = ShapeDataset(csv_path=csv_path, 
                           shape_feat_col_start=shape_feat_col_start, 
                           shape_feat_col_end=shape_feat_col_end,
                           method="resize", label="label", size=size, enable_negatives=enable_negatives, only_roi=only_roi)
    print("len: ", len(dataset))
    
    row = dataset.get_row(idx)
    print("image_path: ", row.get("image_path", "None"))
    print("mask_path: ", row.get("mask_path", "None"))
    
    (x, shape), label = dataset[idx]
    if enable_negatives:
        print(x["positive"].shape, x["negative"].shape, label)
        sitk.WriteImage(sitk.GetImageFromArray(x["positive"]), str(output_dir / f"{idx:03d}_patch_image.nii.gz"))
        sitk.WriteImage(sitk.GetImageFromArray(x["negative"]), str(output_dir / f"{idx:03d}_patch_neg.nii.gz"))
    else:
        print(x.shape, label)
        sitk.WriteImage(sitk.GetImageFromArray(x), str(output_dir / f"{idx:03d}_patch_image.nii.gz"))

    mask_patch = dataset.get_mask_patch(idx)
    sitk.WriteImage(sitk.GetImageFromArray(mask_patch), str(output_dir / f"{idx:03d}_patch_mask.nii.gz"))

if __name__ == '__main__':
    # test1()
    test2()
    # debug()
    ...