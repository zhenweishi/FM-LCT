from collections import defaultdict
import os
from pathlib import Path
from fnmatch import fnmatch
import argparse
from tqdm import tqdm
import SimpleITK as sitk
import pandas as pd
import scipy
import numpy as np
import monai

# parser = argparse.ArgumentParser()
#
#
# parser.add_argument("-s", "--size", type=int, default=48, required=False)
# parser.add_argument("-c", "--crop_methods", type=str, default='all_roi_resize', help="crop methods, e.g. 'crop_1,crop_2'", required=False)
# parser.add_argument("--margin", type=int, default=0, required=False)
# parser.add_argument("--largest", action="store_true", required=False)
#
# args = parser.parse_args()
# if "resize" in 'all_roi_resize' and 48 is None:
#     raise ValueError("Please specify size when using resize method")
# args.crop_methods = 'all_roi_resize'.split(",")

def rglob(path, pattern):
    # pathlib.Path.rglob() does not follow symlinks
    for root, dirs, files in os.walk(path, followlinks=True):
        for filename in files:
            if fnmatch(filename, pattern):
                yield Path(os.path.join(root, filename))

def all_roi_crop(img, roi_center, roi_size, **kwargs):
    crop_size = max(roi_size)
    cropper = monai.transforms.Compose([
        monai.transforms.ToTensor(track_meta=False),
        monai.transforms.EnsureChannelFirst(channel_dim='no_channel'),
        monai.transforms.SpatialCrop(roi_center=roi_center, roi_size=crop_size),
        monai.transforms.SqueezeDim(dim=0),
    ])
    return cropper(img)

def big_roi_resize(img, roi_center, roi_size, tgt_size, **kwargs):
    crop_size = max(max(roi_size), tgt_size)
    cropper = monai.transforms.Compose([
        monai.transforms.ToTensor(track_meta=False),
        monai.transforms.EnsureChannelFirst(channel_dim='no_channel'),
        monai.transforms.SpatialCrop(roi_center=roi_center, roi_size=crop_size),
        monai.transforms.Resize(spatial_size=(tgt_size, tgt_size, tgt_size)) if crop_size > tgt_size else monai.transforms.Lambda(lambda x: x),
        monai.transforms.SqueezeDim(dim=0),
    ])
    return cropper(img)

def all_roi_resize(img, roi_center, roi_size, tgt_size, **kwargs):
    cropper = monai.transforms.Compose([
        monai.transforms.ToTensor(track_meta=False),
        monai.transforms.EnsureChannelFirst(channel_dim='no_channel'),
        monai.transforms.SpatialCrop(roi_center=roi_center, roi_size=roi_size),
        monai.transforms.Resize(spatial_size=(tgt_size, tgt_size, tgt_size)),
        monai.transforms.SqueezeDim(dim=0),
    ])
    return cropper(img)

def crop_img(img, roi_center, tgt_size, **kwargs):
    cropper = monai.transforms.Compose([
        monai.transforms.ToTensor(track_meta=False),
        monai.transforms.EnsureChannelFirst(channel_dim='no_channel'),
        monai.transforms.SpatialCrop(roi_center=roi_center, roi_size=(tgt_size, tgt_size, tgt_size)),
        monai.transforms.SqueezeDim(dim=0),
    ])
    return cropper(img)

def crop_bbox(img, roi_center, roi_size, **kwargs):
    margin = 0
    roi_size = [roi_size[i] + margin * 2 for i in range(3)]
    cropper = monai.transforms.Compose([
        monai.transforms.ToTensor(track_meta=False),
        monai.transforms.EnsureChannelFirst(channel_dim='no_channel'),
        monai.transforms.SpatialCrop(roi_center=roi_center, roi_size=roi_size),
        monai.transforms.SqueezeDim(dim=0),
    ])
    return cropper(img)

def get_roi(mask_np):
    assert mask_np.sum() > 0, "Empty mask"

    # =========================== calculate centroid ===========================
    centroid = scipy.ndimage.center_of_mass(mask_np)
    assert np.isnan(centroid).any() == False

    # =========================== calculate bounding box ===========================
    bbox = monai.transforms.generate_spatial_bounding_box(np.expand_dims(mask_np, axis=0), margin=0, allow_smaller=False)

    return centroid, bbox


def calc(patient_id, mask_path, image_path, output_path):

    crop_row_dict = {}

    image_path = image_path.resolve()
    mask_path = mask_path.resolve()

    image_np = sitk.GetArrayFromImage(sitk.ReadImage(image_path))
    mask_np = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
    mask_np = mask_np.astype("int32")
    mask_np[mask_np != 1] = 0
    if "store_true":
        mask_np = monai.transforms.utils.get_largest_connected_component_mask(mask_np).astype("int32")

    if image_np.shape != mask_np.shape:
        print(" * Shape mismatch: \n    ", image_path, mask_path)
        return None, None

    row = dict(PatientID=patient_id, image_name=image_path.name, mask_name=mask_path.name, image_path=image_path, mask_path=mask_path)
    # =========================== calculate centroid ===========================
    try:
        centroid, bbox = get_roi(mask_np)
    except:
        print(" * Empty mask: ", mask_path)
        return None, None
    x, y, z = centroid
    roi_start, roi_end = bbox

    row["coordX"], row["coordY"], row["coordZ"] = x, y, z
    if roi_start == roi_end:
        print(f"Empty mask: {mask_path}")
        return None, None

    row["roi_start"], row["roi_end"] = roi_start, roi_end
    for i, n in enumerate(["x", "y", "z"]):
        row[f"roi_start_{n}"], row[f"roi_end_{n}"] = row["roi_start"][i], row["roi_end"][i]

    # =========================== crop image ===========================
    roi_center = [(roi_start[i] + roi_end[i]) // 2 for i in range(3)]
    roi_size = [roi_end[i] - roi_start[i] for i in range(3)]
    for method in ['all_roi_resize']:
        patch_img = globals()[method](img=image_np, roi_center=roi_center, roi_size=roi_size, tgt_size=48)
        patch_mask = globals()[method](img=mask_np, roi_center=roi_center, roi_size=roi_size, tgt_size=48)
        method_name = f"{method}{48}" if "resize" in method else method
        method_name = method_name + f"_margin0" if "bbox" in method else method_name
        dst_img_path = Path(output_path).parent / method_name / "image" / Path(mask_path).name # use mask name as image name
        dst_mask_path = Path(output_path).parent / method_name / "mask" / Path(mask_path).name
        dst_img_path = dst_img_path.resolve()
        dst_mask_path = dst_mask_path.resolve()
        dst_img_path.parent.mkdir(parents=True, exist_ok=True)
        dst_mask_path.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(sitk.GetImageFromArray(patch_img), str(dst_img_path))
        sitk.WriteImage(sitk.GetImageFromArray(patch_mask.astype(np.uint8)), str(dst_mask_path))

        # =========================== update row ===========================
        new_row = dict(PatientID=patient_id, image_name=dst_img_path.name, mask_name=dst_mask_path.name, image_path=dst_img_path, mask_path=dst_mask_path)
        new_centroid, new_bbox = get_roi(patch_mask)
        new_row["coordX"], new_row["coordY"], new_row["coordZ"] = new_centroid
        new_row["roi_start"], new_row["roi_end"] = new_bbox
        for i, n in enumerate(["x", "y", "z"]):
            new_row[f"roi_start_{n}"], new_row[f"roi_end_{n}"] = new_row["roi_start"][i], new_row["roi_end"][i]

        crop_row_dict[method_name] = new_row.copy()
        crop_row_dict[method_name]

    return row, crop_row_dict

def preprocessing_main(input_path,input_csv,output):


    img_dir = os.path.join(input_path, 'Image')
    mask_dir = os.path.join(input_path, 'Mask')
    csv_path = Path(output)
    label_path = Path(input_csv) if input_csv else None
    print("=> Image dir: ", img_dir)
    print("=> Mask dir: ", mask_dir)
    print("=> Output csv: ", csv_path)
    if label_path:
        print("=> Label csv: ", label_path)

    img_idx = {path.name: path for path in rglob(img_dir, "*.nii.gz")}
    mask_idx = {path.name: path for path in rglob(mask_dir, "*.nii.gz")}
    has_mask_idx = set(img_idx.keys()) & set(mask_idx.keys())
    label_df = pd.read_csv(label_path) if label_path else None
    print("=> Number of images: ", len(img_idx))
    print("=> Number of masks: ", len(mask_idx))
    print("=> Number of images with mask: ", len(has_mask_idx))
    if label_df is not None:
        print("=> Number of images with label: ", label_df.shape[0])

    img_glob = [img_idx[name] for name in has_mask_idx]
    mask_glob = [mask_idx[name] for name in has_mask_idx]
    if label_df is not None:
        label_idx = {row["Name"]: row for idx, row in label_df.iterrows()}

    data = []
    crop_data = defaultdict(list)
    for img_path, mask_path in tqdm(list(zip(img_glob, mask_glob))):
        row, crop_row_dict = calc(patient_id=img_path.name.replace(".nii.gz", ""), mask_path=mask_path, image_path=img_path, output_path = output)
        if row:
            if label_df is not None:
                old_row = label_idx[img_path.name].copy().to_dict()
                old_row.update(row)
                row = old_row
            data.append(row)
            for method, crop_row in crop_row_dict.items():
                if label_df is not None:
                    old_crop_row = label_idx[img_path.name].copy().to_dict()
                    old_crop_row.update(crop_row)
                    crop_row = old_crop_row
                crop_data[method].append(crop_row)

    df = pd.DataFrame(data)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

    for method, crop_rows in crop_data.items():
        crop_df = pd.DataFrame(crop_rows)
        crop_csv_path = Path(output).parent / method / f"{method}.csv"
        crop_df.to_csv(crop_csv_path, index=False)

