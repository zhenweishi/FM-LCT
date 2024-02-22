import cv2
import numpy as np

def overlay_mask(imgx, maskx, alpha=0.5):
    import numpy as np
    import cv2
    # 如果imgx的深度不是CV_8U, 则转换它
    if imgx.dtype != np.uint8:
        # 根据实际情况进行调整，这里我们将图像值从 16 位转换到 8 位
        imgx = cv2.normalize(imgx, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # 如果maskx的深度不是CV_8U, 则转换它
    if maskx.dtype != np.uint8:
        maskx = cv2.normalize(maskx, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # 将imgx和maskx转换为三通道图像
    imgx_colored = cv2.cvtColor(imgx, cv2.COLOR_GRAY2BGR)
    
    # 创建一个红色遮罩，它具有与imgx相同的尺寸
    red_mask = np.zeros_like(imgx_colored)
    red_mask[:, :, 0] = 255  # 红色通道设为255

    # 使用maskx作为权重来创建半透明的红色遮罩
    overlay = cv2.addWeighted(imgx_colored, 1-alpha, red_mask, alpha, 0, dtype=cv2.CV_8U)
    
    mask3channel = cv2.cvtColor(maskx, cv2.COLOR_GRAY2BGR)
    overlay[mask3channel[:, :, 0] == 0] = imgx_colored[mask3channel[:, :, 0] == 0]

    return overlay