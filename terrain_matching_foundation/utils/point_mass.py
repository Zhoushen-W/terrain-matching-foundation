"""提供点质量滤波的等距网格和高斯过程噪声预测。"""

import numpy as np
from scipy.ndimage import gaussian_filter


def point_mass_grid(radius, step):
    """建立包含中心且不超出给定半径的二维等距偏移网格。

    Args:
        radius: 非负网格半径，单位为像素。
        step: 正的网格间距，单位为像素。

    Returns:
        numpy.ndarray: 形状为 (K, K, 2) 的 float64 偏移，末维为 [x, y]；
            K=2*floor(radius/step)+1，非整除时不额外插入边界节点。
    """
    count = int(np.floor(radius / step))
    axis = np.arange(-count, count + 1, dtype=np.float64) * step
    dx, dy = np.meshgrid(axis, axis)
    return np.stack((dx, dy), axis=-1)


def predict_log_mass(log_mass, process_std, grid_step):
    """在随 INS 平移的网格上用高斯卷积传播归一化概率质量。

    Args:
        log_mass: 二维等距网格上的归一化对数概率质量，零质量用 -inf 表示。
        process_std: 每步每轴的非负过程噪声标准差，单位为像素。
        grid_step: 正的网格间距，单位为像素。

    Returns:
        numpy.ndarray: 同形状的新归一化对数概率质量；零过程噪声时返回副本。
            网格外采用零填充，高斯核截断于四倍标准差并按网格取整。
    """
    if process_std == 0:
        return log_mass.copy()
    mass = gaussian_filter(
        np.exp(log_mass),
        sigma=process_std / grid_step,
        mode="constant",
        cval=0.0,
        truncate=4.0,
    )
    mass /= mass.sum()
    with np.errstate(divide="ignore"):
        return np.log(mass)
