"""提供地形栅格校验和考虑边界及无效像素的双线性采样。"""

import numpy as np

from terrain_matching_foundation.utils.inputs import real_array


def prepare_terrain(terrain_map):
    """创建定位器持有的只读地形副本。

    Args:
        terrain_map: 形状为 (H, W) 的实数 NumPy 栅格，NaN 表示无效区域。

    Returns:
        numpy.ndarray: float64 只读地形副本。

    Raises:
        ValueError: 栅格不是二维或小于 2×2。
    """
    field = real_array(terrain_map, "terrain_map", 2)
    if min(field.shape) < 2:
        raise ValueError("terrain_map must be at least 2x2")
    field = field.copy(order="C")
    field.setflags(write=False)
    return field


def bilinear_sample(field, points):
    """按像素中心坐标向量化采样地形。

    Args:
        field: 已校验的形状为 (H, W) 的地形栅格。
        points: 最后一维为 [x, y] 的坐标数组。

    Returns:
        numpy.ndarray: 与 points 前导维度一致的采样值；越界或正权重
            插值邻点为 NaN 时返回 NaN，零权重邻点不影响结果。
    """
    points = np.asarray(points, dtype=np.float64)
    shape = points.shape[:-1]
    flat = points.reshape(-1, 2)
    height, width = field.shape
    valid = (
        (flat[:, 0] >= 0)
        & (flat[:, 0] <= width - 1)
        & (flat[:, 1] >= 0)
        & (flat[:, 1] <= height - 1)
    )
    result = np.full(len(flat), np.nan, dtype=np.float64)
    xy = flat[valid]
    x0 = np.floor(xy[:, 0]).astype(np.intp)
    y0 = np.floor(xy[:, 1]).astype(np.intp)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    wx = xy[:, 0] - x0
    wy = xy[:, 1] - y0
    weights = np.stack(
        ((1 - wx) * (1 - wy), wx * (1 - wy), (1 - wx) * wy, wx * wy),
        axis=1,
    )
    values = np.stack(
        (field[y0, x0], field[y0, x1], field[y1, x0], field[y1, x1]), axis=1
    )
    result[valid] = np.sum(np.where(weights > 0, values, 0.0) * weights, axis=1)
    return result.reshape(shape)
