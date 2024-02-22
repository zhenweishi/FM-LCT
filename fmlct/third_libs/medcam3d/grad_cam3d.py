import numpy as np
from medcam3d.base_medcam3d import BaseCAM


class GradCAM(BaseCAM):
    def __init__(self, model, target_layers, use_cuda=False,
                 reshape_transform=None):
        super(
            GradCAM,
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
        
        if len(grads.shape) == 5:  # 3d，shape of grads: n, c, z, y, x
            return np.mean(grads, axis=(2, 3, 4))  # shape: n, c
        elif len(grads.shape) == 4:  # 2d，shape of grads: n, c, y, x
            return np.mean(grads, axis=(2, 3))  # shape: n, c
        else:  # shape: (n,)
            return grads  # shape: (n,)
        