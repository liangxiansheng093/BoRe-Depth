import numpy as np
import open3d as o3d
import cv2
from imageio.v2 import imread
from skimage.transform import resize


def depth_to_pointcloud(depth_map, fx, fy, cx, cy):
    """
    Args:
        depth_map: (H x W)
        fx, fy, cx, cy
    Returns:
        (N x 3)
    """
    h, w = depth_map.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth_map
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    return points


def map_rgb_to_pointcloud(depth_map, rgb_image, fx, fy, cx, cy):
    """
    Args:
        depth_map: (H x W)
        rgb_image: (H x W x 3)
        fx, fy, cx, cy

    Returns:
        points: (N x 3)
        colors: (N x 3)
    """
    points = depth_to_pointcloud(depth_map, fx, fy, cx, cy)

    h, w = depth_map.shape
    rgb_image_resized = cv2.resize(rgb_image, (w, h))
    colors = rgb_image_resized.reshape(-1, 3) / 255.0

    return points, colors


# load data
depth_image = np.load("output/factory_02.npy")
rgb_image = cv2.imread("demo/factory.png")
rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

# nyu
# fx, fy, cx, cy = 518.8579, 519.46961, 325.58245, 253.73617
# kitti
# fx, fy, cx, cy = 483.3489, 492.5697, 408.33602, 118.00166
# iBims-1
fx, fy, cx, cy = 559.62, 558.14, 361.87, 241.99


if depth_image.shape[0] != rgb_image.shape[0] or depth_image.shape[1] != rgb_image.shape[1]:
    depth_map = resize(depth_image, (rgb_image.shape[0], rgb_image.shape[1]), anti_aliasing=True)

# remove invalid pixels
y1, y2, x1, x2 = 10, 471, 10, 630
depth_image = cv2.resize(depth_image, (640, 480), interpolation=cv2.INTER_LINEAR)
mask = np.zeros_like(depth_image)
mask[y1:y2, x1:x2] = 1
depth_image = depth_image * mask

points, colors = map_rgb_to_pointcloud(depth_image, rgb_image, fx, fy, cx, cy)
# Open3D
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)
rotation_matrix = pcd.get_rotation_matrix_from_xyz((np.pi, 0, 0))
pcd.rotate(rotation_matrix)
# vis
o3d.visualization.draw_geometries([pcd], window_name="Colored Point Cloud")
