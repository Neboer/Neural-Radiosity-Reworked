import numpy as np
from openexr_numpy import imread, imwrite


def create_direction_map(posw, camera_pos):
    """
    根据像素点的实际坐标矩阵 posw 和相机位置 camera_pos，生成一个 x*y*2 的极坐标方向矩阵。
    如果某点的坐标为 (0, 0, 0)，则返回 (0, 0)。

    参数：
    - posw: 一个形状为 (x, y, 3) 的 numpy 数组，表示像素点的实际坐标。
    - camera_pos: 一个长度为 3 的 numpy 数组，表示相机的实际坐标 (x, y, z)。

    返回：
    - 一个形状为 (x, y, 2) 的 numpy 数组，表示每个点到相机的方向 (φ, θ)。
    """
    # 计算到相机的方向向量
    direction = camera_pos - posw  # (x, y, 3)

    # 计算距离（模长）
    distances = np.linalg.norm(direction, axis=-1)  # (x, y)

    # 处理空点：找到 posw 为 (0, 0, 0) 的点
    empty_mask = np.all(posw == 0, axis=-1)  # (x, y)

    # 计算极坐标方向
    with np.errstate(divide='ignore', invalid='ignore'):  # 忽略除零警告
        phi = np.arctan2(direction[..., 1], direction[..., 0])  # 方位角 φ
        theta = np.arccos(direction[..., 2] / distances)  # 俯仰角 θ

    # 将空点设置为 0
    phi[empty_mask] = 0
    theta[empty_mask] = 0

    # 堆叠结果为 (x, y, 2)
    direction_map = np.stack((phi, theta), axis=-1)  # (x, y, 2)

    return direction_map


class DataSlice:
    def __init__(self, camera_pos: np.array, posw_file_path, color_file_path=None):
        self.camera_pos = camera_pos
        self.posw = imread(posw_file_path)
        self.color = imread(color_file_path) if color_file_path else np.zeros_like(self.posw)
        self.twod_mask = np.all(self.posw == [0.0, 0.0, 0.0], axis=-1)
        self._oned_mask = None
        self.color[self.twod_mask] = 0
        self.direction_map = create_direction_map(self.posw, self.camera_pos)
        self.collapse()

    def print_dimensions(self):
        print(f"posw shape: {self.posw.shape}")
        print(f"color shape: {self.color.shape}")
        print(f"direction shape: {self.direction_map.shape}")

    # 将posw, direction_map, color全部转换成一维信息，同时清空所有空点，用于训练，但我们保留空点的位置。
    def collapse(self):
        posw = self.posw.reshape(-1, 3)  # shape: (x * y, 3)
        direction = self.direction_map.reshape(-1, 2)  # shape: (x * y, 2)
        color = self.color.reshape(-1, 3)  # shape: (x * y, 3)
        # print("data concatenated")
        # print(f"shape: {posw.shape}")

        # 移除所有零点
        # 生成一个布尔掩码，标记 posw 中不是 (0, 0, 0) 的点
        self._oned_mask = ~(np.all(posw == 0, axis=-1))

        # 使用掩码筛选有效点
        posw = posw[self._oned_mask]
        direction = direction[self._oned_mask]
        color = color[self._oned_mask]
        # print("zero points removed")
        # print(f"shape: {posw.shape}")

        self.posw_collapsed = posw
        self.direction_collapsed = direction
        self.color_collapsed = color
    
    # 将预测的颜色值重新填充到原始的图像中，与collapse相反
    def expand(self, predicted_color):
        color = np.zeros_like(self.color).reshape(-1, 3)
        color[self._oned_mask] = predicted_color
        return color.reshape(self.color.shape)