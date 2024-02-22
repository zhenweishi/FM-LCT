import os.path
import sys
import argparse

current_path = os.getcwd()
delete_part = current_path.split('/')[-1]
current_path = current_path.replace('/'+delete_part, '')
sys.path.append(current_path)
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
from pathlib import Path
import pandas as pd
from fmlct.lib.models import vit_3d_base_patchsize8
from fmlct.lib.utils import read_array
import torch
import monai
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from umap.umap_ import UMAP
from sklearn import preprocessing



DX_MAP = {f"d{i}": f"d{i}" for i in range(10)}
DX_MAP.update({"bl": "d0", "1c": "d1", "2c": "d2", "po": "d3"})

def pid_dx_cx_parser(name):
    try:
        center, dx, cx, pid = name.split('_')
    except:
        raise Exception(f"=> Failed to parse name:{name}")
    dx = DX_MAP[dx]
    return pid, dx, cx

def z_score(x):
    return (x - x.mean()) / (x.std() + 1e-8)

def all_roi_resize(img, roi_center, roi_size, tgt_size, **kwargs):
    # crop_size = max(roi_size)
    cropper = monai.transforms.Compose([
        monai.transforms.ToTensor(track_meta=False),
        monai.transforms.EnsureChannelFirst(channel_dim='no_channel'),
        monai.transforms.SpatialCrop(roi_center=roi_center, roi_size=roi_size),
        monai.transforms.Resize(spatial_size=(tgt_size, tgt_size, tgt_size)),
        monai.transforms.SqueezeDim(dim=0),
    ])
    return cropper(img)

def calc_bbox(mask_np):
    # =========================== calculate bounding box ===========================
    bbox = monai.transforms.generate_spatial_bounding_box(np.expand_dims(mask_np, axis=0), margin=0, allow_smaller=False)
    roi_start, roi_end = bbox

    if roi_start == roi_end:
        return False, None, None

    roi_center = [(roi_start[i] + roi_end[i]) // 2 for i in range(3)]
    roi_size = [roi_end[i] - roi_start[i] for i in range(3)] 
    return True, roi_center, roi_size

def read_from_memory(img_idx, pid, dx, cx, size):
    """img_idx: # key: (pid, dx, cx), value: dict(ok, roi_center, roi_size, mask_patch, image_path, image_patch)
    """
    img_dict = img_idx[(pid, dx, cx)]
    if not img_dict["ok"]:
        return None
    if img_dict.get("image_patch", None) is None:
        img = read_array(img_dict["image_path"])
        img = all_roi_resize(img, img_dict["roi_center"], img_dict["roi_size"], size).numpy()
        img_idx[(pid, dx, cx)]["image_patch"] = img
    return img_idx[(pid, dx, cx)]["image_patch"]
    

def dx_pca(dx_feat_df, tgt_cx):
    tgt_idx = (dx_feat_df.msg == "ok") & (dx_feat_df.cx == tgt_cx)
    cx_feat_df = dx_feat_df[tgt_idx].reset_index(drop=True)

    tgt_values = cx_feat_df["feature"].values
    tgt_values = np.concatenate(tgt_values, axis=0)
    if tgt_values.shape[0] >= 16:
        dim = 16
    else:
        dim = int(0.8 * min(tgt_values.shape))
        print(f'Warning: Dimension of PCA must be between 0 and min(n_samples, n_features)!')
    pca_feat = PCA(n_components=dim).fit_transform(tgt_values)
    pca_feat_df = pd.DataFrame(pca_feat).rename(columns={i: f"feat_{i}" for i in range(dim)})
    pca_feat_df = pd.concat([cx_feat_df[["center", "pid", "dx", "cx"]], pca_feat_df], axis=1)
    pca_feat_df.rename(columns={"pid": "ID"}, inplace=True)
    return pca_feat_df


def dx_worker(img_idx, mask_idx, model, img_root, mask_root, output_path, roi_only, size, tgt_cx_list):
    model.eval()
    # Read all mask -> calc roi -> load mask_patches
    print("=> Reading mask...")
    mask_list = os.listdir(mask_root)
    mask_list = [os.path.join(mask_root, i) for i in mask_list]
    img_list = os.listdir(img_root)
    img_list = [os.path.join(img_root, i) for i in img_list]
    mask_list.sort()
    img_list.sort()
    for mask_path in tqdm(mask_list):
        name = mask_path.split('/')[-1].split('.')[0]
        center = name.split('_')[0]
        pid, dx, cx = pid_dx_cx_parser(name)

        if mask_idx.get((pid, dx), None):
            continue

        mask = read_array(mask_path).astype(np.uint8)
        mask[mask > 0] = 1
        ok, roi_center, roi_size = calc_bbox(mask)
        mask_patch = all_roi_resize(mask, roi_center, roi_size, size).numpy().astype(np.uint8) if ok else None

        mask_idx[(pid, dx)] = dict(ok=ok, roi_center=roi_center, roi_size=roi_size, mask_patch=mask_patch)

    # Create image dataframe and index of image path
    data = [] # tmp, for creating dataframe
    for img_path in tqdm(img_list):
        name = img_path.split('/')[-1].split('.')[0]
        center = name.split('_')[0]
        pid, dx, cx = pid_dx_cx_parser(name)
        
        if img_idx.get((pid, dx, cx), None) is None:
            mask_dict = mask_idx.get((pid, dx), None)
            if mask_dict is None:
                print(f"=> {pid} {dx} has no mask")
                continue
            mask_dict["image_path"] = img_path
            img_idx[(pid, dx, cx)] = mask_dict.copy()
        data.append(dict(pid=pid, dx=dx, cx=cx, img_path=img_path))
    df = pd.DataFrame(data)
    # df.to_csv("test.csv")

    # Keep only target cx
    std_cx_list = []
    for tgt_cx in tgt_cx_list:
        if "-" in tgt_cx:
            cx_1, cx_2 = tgt_cx.split('-')
            std_cx_list.append(cx_1)
            std_cx_list.append(cx_2)
        else:
            std_cx_list.append(tgt_cx)
    std_cx_list = list(set(std_cx_list))
    df = df[df.cx.isin(std_cx_list)]

    # Read image and calc feature
    data = []
    cnt = 0
    print("=> Reading image and calc feature...")
    for pid in tqdm(df.pid.unique(), desc=f"{center}-{dx}", total=len(df.pid.unique())):
        # cnt += 1
        # if cnt == 10:
        #     break

        if not mask_idx[(pid, dx)]["ok"]:
            data.append(dict(center=center, pid=pid, dx=dx, cx=tgt_cx, feature=None, msg="no mask"))
            continue

        for tgt_cx in tgt_cx_list:
            try:
                if "-" in tgt_cx:
                    cx_1, cx_2 = tgt_cx.split('-')
                    img_1 = read_from_memory(img_idx, pid, dx, cx_1, size)
                    img_2 = read_from_memory(img_idx, pid, dx, cx_2, size)
                    img = img_1 - img_2
                    # import SimpleITK as sitk
                    # sitk.WriteImage(sitk.GetImageFromArray(img_1), "img_1.nii.gz")
                    # sitk.WriteImage(sitk.GetImageFromArray(img_2), "img_2.nii.gz")
                    # sitk.WriteImage(sitk.GetImageFromArray(img), "sub.nii.gz")
                    # import sys
                    # sys.exit()
                else:
                    img = read_from_memory(img_idx, pid, dx, tgt_cx, size)

                if roi_only:
                    mask = mask_idx[(pid, dx)]["mask_patch"]
                    mask[mask > 0] = 1
                    if img.min() < 0:
                        # background = np.full_like(img, -1024) # CT
                        background = np.full_like(img, 0) # sub
                    else:
                        background = np.zeros_like(img) # MR
                    img = np.where(mask == 1, img, background)

                img = torch.from_numpy(img.astype(np.float32)).unsqueeze(0).unsqueeze(0).float().cuda()
                img = z_score(img)
                
                with torch.no_grad():
                    output = model.forward_features(img).detach().cpu().numpy() # (1, N, 768)
                    # Get the cls_token, -> 1, 768
                    output = output[:, 0, :]
                    data.append(dict(center=center, pid=pid, dx=dx, cx=tgt_cx, feature=output, msg="ok"))
            except Exception as e:
                data.append(dict(center=center, pid=pid, dx=dx, cx=tgt_cx, feature=None, msg=str(e)))
                continue
    

    feat_df = pd.DataFrame(data)

    print("=> Saving...")
    out_dir = Path(output_path)
    og_csv_path = out_dir / ("feture_csv_" + ("roi_only" if roi_only else "bbox")) / f"{center}_{dx}_{tgt_cx}.csv"
    og_csv_path.parent.mkdir(parents=True, exist_ok=True)
    _ = np.concatenate(feat_df["feature"].values, axis=0)
    og_feat_df = pd.DataFrame(_).rename(columns={i: f"feat_{i}" for i in range(_.shape[-1])})
    og_feat_df = pd.concat([feat_df[["center", "pid", "dx", "cx"]], og_feat_df], axis=1)
    og_feat_df.rename(columns={"pid": "ID"}, inplace=True)
    og_feat_df.to_csv(og_csv_path, index=False)

    print("=> PCA...")
    for tgt_cx in tgt_cx_list:
        try:
            cx_pca_feat_df = dx_pca(feat_df, tgt_cx)
            if roi_only==True:
                csv_path = os.path.join(out_dir, "feture_csv_roi_only")
            else:
                csv_path = os.path.join(out_dir, "feture_csv_bbox")
            if not os.path.exists(os.path.join(out_dir, "feture_csv_roi_only")):
                os.makedirs(os.path.join(out_dir, "feture_csv_roi_only"))
            if not os.path.exists(os.path.join(out_dir, "feture_csv_bbox")):
                os.makedirs(os.path.join(out_dir, "feture_csv_bbox"))
            cx_pca_feat_df.to_csv(csv_path+'/'+f"{center}_{dx}_{tgt_cx}_pca.csv", index=False)
            id_list, feature_list = read_feature(csv_path+'/'+f"{center}_{dx}_{tgt_cx}_pca.csv", ['ID'])
            dim_reduction(feature_list, csv_path.replace(f"/{center}_{dx}_{tgt_cx}_.csv", ""))
        except Exception as e:
            print(f"{'roi_only' if roi_only else ''} {center} {dx} {tgt_cx} failed in PCA: ", e)


def dim_reduction(feature_list, output_path, n=2, random=66):
    tsne = TSNE(n_components=n, random_state=random, init='pca', learning_rate='auto')
    feature_embedded = tsne.fit_transform(feature_list)
    plt.scatter(feature_embedded[:, 0], feature_embedded[:, 1], s=5)
    plt.title('t-SNE Visualization')
    plt.savefig(os.path.join(output_path, 'TSNE.png'))
    plt.clf()
    umap = UMAP(n_components=n, random_state=random)
    feature_embedded = umap.fit_transform(feature_list)
    plt.scatter(feature_embedded[:, 0], feature_embedded[:, 1], s=5)
    plt.title('UMAP Visualization')
    plt.savefig(os.path.join(output_path, 'UMAP.png'))
    plt.close() #plt.clf()


def read_feature(path, id_col, feature_list=['ALL']):
    if feature_list == ['ALL']:
        feature_id = []
        if path.endswith('xlsx'):
            df = pd.read_excel(path)
            keys_list = list(df.keys())
            for key in keys_list:
                if key.startswith('feat'):
                    feature_id.append(key)
            id_list = np.array(pd.read_excel(path, usecols=id_col)).squeeze()
            # feature_list = np.array(pd.read_excel(path, usecols=[feature_list])).squeeze()
            feature_list = np.array(pd.read_excel(path, usecols=feature_id)).squeeze()
        elif path.endswith('csv'):
            df = pd.read_csv(path)
            keys_list = list(df.keys())
            for key in keys_list:
                if key.startswith('feat'):
                    feature_id.append(key)
            id_list = np.array(pd.read_csv(path, usecols=id_col)).squeeze()
            feature_list = np.array(pd.read_csv(path, usecols=feature_id)).squeeze()
        else:
            raise ValueError("Only support .xlsx or .csv file.")


    else:
        if path.endswith('xlsx'):
            df = pd.read_excel(path)
            keys_list = list(df.keys())
            keys_list.remove(id_col)

            keys_list.remove('pid')

            id_list = np.array(pd.read_excel(path, usecols=[id_col])).squeeze()
            feature_list = np.array(pd.read_excel(path, usecols=keys_list)).squeeze()
        elif path.endswith('csv'):
            df = pd.read_csv(path)
            keys_list = list(df.keys())
            keys_list.remove(id_col)
            id_list = np.array(pd.read_excel(path, usecols=[id_col])).squeeze()
            feature_list = np.array(pd.read_excel(path, usecols=keys_list)).squeeze()
        else:
            raise ValueError("Only support .xlsx or .csv file.")
    min_max_scaler = preprocessing.MinMaxScaler()
    min_max_scaler.fit(feature_list)
    feature_list = min_max_scaler.transform(feature_list)
    return id_list, feature_list


def load_model(pretrain):
    model = vit_3d_base_patchsize8(img_size=48, in_chans=1)

    # load pretrained weights
    print(f"=> Start loading pretrained weights from {pretrain}")
    checkpoint = torch.load(pretrain, map_location='cpu')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    for k in list(state_dict.keys()):
        # retain only base_encoder up to before the embedding layer
        if k.startswith('base_encoder.'):
            # remove prefix
            state_dict[k[len("base_encoder."):]] = state_dict[k]
            del state_dict[k] 
        if k.startswith('head.'):
            if model.head.weight.shape != state_dict[k].shape:
                del state_dict[k]                
        
    msg = model.load_state_dict(state_dict, strict=False)
    print(f'Loading messages: \n {msg}')
    print(f"=> Finish loading pretrained weights from {pretrain}")
    return model.cuda()
 
def extraction_main(dataset_path, ckpt_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    model = load_model(ckpt_path)
    img_root = os.path.join(dataset_path, "Image")
    mask_root = os.path.join(dataset_path, "Mask")
    img_idx = {} # key: (pid, dx, cx), value: dict(ok, roi_center, roi_size, mask_patch, image_path, image_patch)
    mask_idx = {} # key: (pid, dx), value: dict(ok, roi_center, roi_size, mask_patch)
    for roi_only in [True, False]:
        print("=> Processing", "roi_only" if roi_only else "")
        dx_worker(img_idx, mask_idx, model, img_root, mask_root, output_path, roi_only=roi_only, size=48, tgt_cx_list=['C2'])

