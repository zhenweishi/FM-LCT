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
from tqdm import tqdm
monai.data.set_track_meta(False)
import time

def get_total_num_patches(image, window_size, overlap):
    D, H, W = image.shape
    window_depth, window_height, window_width = window_size
    step_depth, step_height, step_width = (int(window_depth * (1 - overlap)), int(window_height * (1 - overlap)), int(window_width * (1 - overlap)))

    # Calculate the number of patches in each dimension
    patches_d = D // step_depth + 1
    patches_h = H // step_height + 1
    patches_w = W // step_width + 1

    # Calculate total number of patches
    total_patches = patches_d * patches_h * patches_w
    return total_patches

def get_padded_patch_from_3d_volume(image, idx, window_size, overlap, padding_value=0, stay_in_volume=False):
    """
    Extract a single patch from a 3D volume with channels using a sliding window approach,
    with padding if the patch exceeds the boundaries of the volume.

    Parameters:
    - img: A 4D tensor of shape (C, D, H, W), where C is the number of channels,
           D is depth, H is height, and W is width.
    - window_size: A tuple of 3 integers (window_depth, window_height, window_width).
    - step_size: A tuple of 3 integers (step_depth, step_height, step_width).
    - idx: The index of the patch to extract.
    - padding_value: The value to use for padding if the patch exceeds the volume boundaries.

    Returns:
    - A tensor containing the patch at the specified index, with padding if necessary.
    """
    # Unpack the dimensions of the img and the window size
    D, H, W = image.shape
    window_depth, window_height, window_width = window_size
    # print("image.shape: ", image.shape)
    # print("window_size: ", window_size)
    step_depth, step_height, step_width = (int(window_depth * (1 - overlap)), int(window_height * (1 - overlap)), int(window_width * (1 - overlap)))

    # Calculate the number of patches in each dimension
    patches_d = D // step_depth + 1
    patches_h = H // step_height + 1
    patches_w = W // step_width + 1

    # Calculate total number of patches
    total_patches = patches_d * patches_h * patches_w

    # Check if the index is within the range of total patches
    if idx < 0 or idx >= total_patches:
        raise IndexError("Index out of range")

    # Calculate the patch's starting position in each dimension
    idx_d = (idx // (patches_h * patches_w)) * step_depth
    idx_h = ((idx % (patches_h * patches_w)) // patches_w) * step_height
    idx_w = ((idx % (patches_h * patches_w)) % patches_w) * step_width

    # 计算实际可以从图像中提取的块的大小
    actual_depth = min(window_depth, D - idx_d)
    actual_height = min(window_height, H - idx_h)
    actual_width = min(window_width, W - idx_w)
    ratio = actual_depth * actual_height * actual_width / (window_depth * window_height * window_width)

    if stay_in_volume:
        # Calculate the patch's ending position in each dimension
        end_d = min(idx_d + window_depth, D)
        end_h = min(idx_h + window_height, H)
        end_w = min(idx_w + window_width, W)
        start_d = end_d - window_depth
        start_h = end_h - window_height
        start_w = end_w - window_width

        sub_block = image[start_d:end_d, start_h:end_h, start_w:end_w]
        return sub_block, ratio
        
    else:
        # 创建一个与期望块大小相同的零数组
        patch = np.full(window_size, padding_value).astype(image.dtype)

        # 提取子块
        sub_block = image[idx_d:idx_d + actual_depth, idx_h:idx_h + actual_height, idx_w:idx_w + actual_width]

        # 将子块填充到 patch 中
        patch[:actual_depth, :actual_height, :actual_width] = sub_block
        ratio = actual_depth * actual_height * actual_width / (window_depth * window_height * window_width)

    return patch, ratio

def repack_patches(patches, original_shape, window_size, overlap, threshold=0.5):
    """
    Reconstruct a 3D volume from its patches, considering padding.

    Parameters:
    - patches: List of 3D numpy arrays representing the patches.
    - original_shape: The shape of the original 3D volume (D, H, W).
    - window_size: A tuple of 3 integers (window_depth, window_height, window_width) for patch size.
    - overlap: Overlap fraction used while extracting patches.

    Returns:
    - Reconstructed 3D volume.
    """

    # Unpack the dimensions of the img and the window size
    D, H, W = original_shape
    window_depth, window_height, window_width = window_size
    step_depth, step_height, step_width = (int(window_depth * (1 - overlap)), int(window_height * (1 - overlap)), int(window_width * (1 - overlap)))

    # Calculate the number of patches in each dimension
    patches_d = D // step_depth + 1
    patches_h = H // step_height + 1
    patches_w = W // step_width + 1

    # Initialize an empty array for the reconstructed image
    reconstructed_image = np.zeros((window_depth * patches_d, window_height * patches_h, window_width * patches_w))

    # Iterate over each patch
    for idx, (patch, ratio) in enumerate(patches):
        # Calculate the patch's starting position in each dimension
        idx_d = (idx // (patches_h * patches_w)) * step_depth
        idx_h = ((idx % (patches_h * patches_w)) // patches_w) * step_height
        idx_w = ((idx % (patches_h * patches_w)) % patches_w) * step_width

        if ratio < threshold:
            continue
        reconstructed_image[idx_d:idx_d + window_depth, idx_h:idx_h + window_height, idx_w:idx_w + window_width] = patch

    reconstructed_image = reconstructed_image[:D, :H, :W]

    return reconstructed_image

class SlidingDataset(datasets.PatchDataset):
    def __init__(self, overlap=0.0, padding_value=0, stay_in_volume=True, patch_idx_json_path=None, patch_idx_save_path=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.method = None
        self._patch_idx = {} # belongs to which volume
        self._num_volumes = self._len
        self._num_patches = 0

        # if self.read_in_memory:
        #     print("=> Loading images and masks into memory...")
        #     for index in tqdm(range(self._num_volumes)):
        #         row = self._rows[index]
        #         # Read Image
        #         img = self.read_image(row.image_path)
        #         mask = self.read_mask(row.mask_path) if self.only_roi or self.get_with_mask else None

        #         # Get Positive Patch
        #         pos_patch, mask_patch = self.get_positive_patch(index, img, mask)
        #         if self.read_in_memory:
        #             self.cookies[index] = (pos_patch, mask_patch)
        print("=> Testing speed of loading ten images...")
        start_time = time.perf_counter()
        for index in tqdm(range(10)):
            row = self._rows[index]
            # Read Image
            img = self.read_image(row.image_path)
            mask = self.read_mask(row.mask_path) if self.only_roi or self.get_with_mask else None
        time_cost = time.perf_counter() - start_time
        print(f"Time: {time_cost:.2f}s / 10 images")
        print("Read all images may take: ", f"{time_cost * self._num_volumes / 60 / 10:.2f}", "mins") 


        # calculate the number of patches for each volume, update self._patch_idx
        if patch_idx_json_path is not None:
            print("=> Loading patch index from json file...")
            import json
            with open(patch_idx_json_path, 'r') as f:
                self._patch_idx = json.load(f)
                # key: str to int
                self._patch_idx = {int(k): v for k, v in self._patch_idx.items()}
            self._num_patches = len(self._patch_idx)
        else:
            print("=> Calculating the number of patches for each volume...")
            for vol_idx in tqdm(range(self._num_volumes)):
                vol, _ = super().__getitem__(vol_idx, transform=False)
                num_patches = get_total_num_patches(vol, self.size, overlap)
                for patch_idx in range(self._num_patches, self._num_patches + num_patches):
                    self._patch_idx[patch_idx] = (vol_idx, patch_idx - self._num_patches)
                self._num_patches += num_patches
        self._len = self._num_patches

        self.overlap = overlap
        self.padding_value = padding_value
        self.stay_in_volume = stay_in_volume
        if patch_idx_save_path:
            self.save_patch_idx(patch_idx_save_path)

    def save_patch_idx(self, json_path):
        import json
        with open(json_path, 'w') as f:
            json.dump(self._patch_idx, f)

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        vol_idx, inner_patch_idx = self._patch_idx[index]
        row = self._rows[vol_idx]
        vol, label = super().__getitem__(vol_idx, transform=False)
        try:
            pos_patch, ratio = get_padded_patch_from_3d_volume(image=vol, 
                                                        idx=inner_patch_idx, 
                                                        window_size=self.size, 
                                                        overlap=self.overlap, 
                                                        padding_value=self.padding_value, 
                                                        stay_in_volume=self.stay_in_volume)
        except Exception as e:
            print("vol_idx: ", vol_idx, "inner_patch_idx: ", inner_patch_idx, "vol.shape: ", vol.shape, "self.size: ", self.size, e)
            pos_patch = self.resize_img(vol)

        pos_patch = self.transform(pos_patch) if self.transform else pos_patch
        # Get Label
        try:
            target = int(row[self.label]) if self.label is not None else False
        except:
            target = False
        
        # Get Negative Patch
        if self.enable_negatives:
            raise NotImplementedError
        
        return pos_patch, target
    
    def get_mask_patch(self, index):
        vol_idx, inner_patch_idx = self._patch_idx[index]
        vol = super().get_mask_patch(vol_idx)
        patch, ratio = get_padded_patch_from_3d_volume(image=vol, 
                                                       idx=inner_patch_idx, 
                                                       window_size=self.size, 
                                                       overlap=self.overlap, 
                                                       padding_value=self.padding_value, 
                                                       stay_in_volume=self.stay_in_volume)
        patch[patch > 0] = 1
        return patch

def test2():
    from pathlib import Path
    import shutil

    csv_path = "/media/wzt/wdc18t/ProcessedData/ROI/crop_bbox_margin24/R.csv"
    save_path = "/media/wzt/wdc18t/ProcessedData/ROI/crop_bbox_margin24/R_overlap05.json"
    enable_negatives = False
    size = 48
    only_roi = False
    idx = 25
    overlap = 0.5

    output_dir = Path("/mnt/tmp/0")
    output_dir.mkdir(exist_ok=True, parents=True)  

    dataset = SlidingDataset(csv_path=csv_path, label="label", size=size, enable_negatives=enable_negatives, only_roi=only_roi, patch_idx_save_path=save_path, overlap=overlap)
    print("len: ", len(dataset))
    
    # row = dataset.get_row(idx)
    # print("image_path: ", row.get("image_path", "None"))
    # print("mask_path: ", row.get("mask_path", "None"))
    
    x, label = dataset[idx]
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