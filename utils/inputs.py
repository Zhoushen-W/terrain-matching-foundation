"""校验像素轨迹与观测输入并统一窗口输出长度。"""

import numpy as np


def real_array(value, name, ndim):
    """校验实数 NumPy 数组并转换计算精度。

    Args:
        value: 输入数组。
        name: 用于错误信息的字段名称。
        ndim: 要求的维度数。

    Returns:
        numpy.ndarray: float64 数组，可能与输入共享内存。

    Raises:
        ValueError: 输入类型或维度不合法。
    """
    if not isinstance(value, np.ndarray):
        raise ValueError(f"{name} must be a NumPy array")
    if value.ndim != ndim or value.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a {ndim}D real numeric array")
    return np.asarray(value, dtype=np.float64)


def validate_trajectories(reference, ins, depths):
    """校验轨迹与观测，参考轨迹数值不参与定位计算。

    Args:
        reference: 仅用于外部评估的参考轨迹，形状为 (N, 2)。
        ins: INS 轨迹，形状为 (N, 2)。
        depths: 深度观测，形状为 (N,)。

    Returns:
        tuple: float64 INS 轨迹和深度观测。

    Raises:
        ValueError: 输入形状不一致或存在非有限值。
    """
    reference = real_array(reference, "reference_trajectory", 2)
    ins = real_array(ins, "ins_trajectory", 2)
    depths = real_array(depths, "depth_observations", 1)
    if ins.shape[1] != 2 or reference.shape != ins.shape:
        raise ValueError("Both trajectories must have the same shape (N, 2)")
    if depths.shape != (len(ins),):
        raise ValueError("depth_observations must have shape (N,)")
    for name, array in (("reference", reference), ("ins", ins), ("depths", depths)):
        if not np.isfinite(array).all():
            raise ValueError(f"{name} must contain only finite values")
    return ins, depths


def window_starts(length, window_size):
    """产生步长为一的完整窗口起点。

    Args:
        length: 输入轨迹点数。
        window_size: 窗口长度。

    Returns:
        range: 同时代表有效输出索引的窗口起点。
    """
    return range(max(0, length - window_size + 1))
