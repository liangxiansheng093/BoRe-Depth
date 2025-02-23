import torchvision.transforms as T
import numpy as np
import cv2
from PIL import Image


def visualize_image(image):
    """
    image: (3, H, W)
    """
    x = (image.cpu() * 0.225 + 0.45)
    return x


def visualize_depth(depth, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """
    x = depth.cpu().numpy()
    x = np.nan_to_num(x)
    mi = np.min(x)
    ma = np.max(x)
    x = (x-mi)/(ma-mi+1e-8)
    x = (255*x).astype(np.uint8)
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    x_ = T.ToTensor()(x_)
    return x_
