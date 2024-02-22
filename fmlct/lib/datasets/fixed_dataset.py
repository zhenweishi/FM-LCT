import torch
import torch.nn as nn

import sys
sys.path.append("/home/wzt/src/wzt_framework")
from fmlct.lib.utils.med_image import read_array, write_array

class ROIConv3d(torch.nn.Module):
    def __init__(self, kernel_size=5):
        super(ROIConv3d, self).__init__()
        # 在这里定义您的自定义卷积核权重
        self.padding = kernel_size // 2  # 计算填充大小以保持输入和输出形状一致
        self.weight = torch.ones(1, 1, kernel_size, kernel_size, kernel_size)

    def forward(self, x):
        # 在前向传播中使用自定义卷积核进行卷积操作，并使用合适的填充来保持输入形状不变
        return torch.nn.functional.conv3d(x, self.weight, padding=self.padding)
    
    def cuda(self):
        self.weight = self.weight.cuda()
        return self

mask = read_array('/media/wzt/wdc18t/ProcessedData/BR_benign_malignant/ROI/crop_bbox_margin24/mask/SY-HXM_d0_C2_10099473_left.nii.gz')
mask_tensor = torch.from_numpy(mask).float().cuda()
roi_conv3d = ROIConv3d(kernel_size=11).cuda()
roi_mask = roi_conv3d(mask_tensor[None][None])
# write_array("/mnt/tmp/0.nii.gz", roi_mask.cpu().numpy())

def rank_x(x, p_list, v_list):
    """
    demo:
    x = torch.arange(27).view(3, 3, 3) + 1
    p_list = [  95,   90,   85,   80,   75]
    v_list = [ 1.0,  2.0,  3.0,  4.0,  5.0]
    """
    xf = x.flatten()
    xf = xf[xf > 0]
    x_sorted = torch.sort(xf).values
    n = x_sorted.numel()
    p_list = torch.tensor([x_sorted[int(p * n)] for p in p_list])
    
    def _dfs(i):
        if i == len(p_list):
            return torch.zeros_like(x)
        else:
            return torch.where(p_list[i] <= x, v_list[i], _dfs(i+1))

    return _dfs(i=0).to(x.device)

roi_mask_rank = rank_x(roi_mask[0, 0], [  0.95,   0.90,   0.87,   0.85,   0.80], [ 5.0,  4.0,  3.0,  2.0,  1.0])
write_array("/mnt/tmp/1.nii.gz", roi_mask_rank.cpu().numpy())

def sample_from_each_label(roi_mask_rank):
    samples = []
    for label in range(6):  # 遍历每个标签
        label_indices = torch.nonzero(roi_mask_rank == label, as_tuple=False)
        if len(label_indices) > 0:  # 确保该标签存在
            random_index = torch.randint(len(label_indices), (1,), device='cuda')
            sample = label_indices[random_index]
            samples.append(sample)
        else:
            samples.append(None)
    return samples

points = sample_from_each_label(roi_mask_rank)
for label in range(len(points)):
    if points[label] is not None:
        print(f'Label {label} has point {points[label]}')
    else:
        print(f'Label {label} has no points')
