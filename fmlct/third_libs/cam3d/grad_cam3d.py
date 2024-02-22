import numpy as np
from .base_cam3d import BaseCAM3D


class GradCAM3D(BaseCAM3D):
    def __init__(self, model, target_layers, use_cuda=False,
                 reshape_transform=None):
        super(
            GradCAM3D,
            self).__init__(
            model,
            target_layers,
            use_cuda,
            reshape_transform)

    def get_cam_weights(self,
                        input_tensor,
                        target_layer,
                        target_category,
                        activations,
                        grads):
        assert len(grads.shape) == 5 # 3d，shape of grads: n, c, z, y, x
        return np.mean(grads, axis=(2, 3, 4))
