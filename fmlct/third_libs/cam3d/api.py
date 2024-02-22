from fmlct.third_libs.cam3d import GradCAM3D
from fmlct.third_libs.cam3d.utils.model_targets import ClassifierOutputTarget
from fmlct.third_libs.cam3d.utils.image import show_cam_on_image
import timm
import numpy as np

def run_vit_cam(model, input_tensor, target=1, invert=True, clip=0.):
    # =================== grad-cam vit 3D ===================
    target_layers = [model.blocks[-1].norm1]
    def reshape_transform_3d(tensor):
        tensor = tensor[:, 1 :  , :]
        height, width, depth = timm.layers.to_3tuple(int(np.round(np.power(tensor.shape[1], 1/3))))
        result = tensor.reshape(tensor.size(0), height, width, depth, tensor.size(-1))
        result = result.permute(0, 4, 1, 2, 3)
        return result
    target_func = [ClassifierOutputTarget(target)]

    cam = GradCAM3D(model=model, target_layers=target_layers, reshape_transform=reshape_transform_3d)
    cam_img = cam(input_tensor=input_tensor, targets=target_func)
    del cam

    out = cam_img[0]
    if invert:
        out = -out + 1
    out = out.clip(clip, out.max())

    return out