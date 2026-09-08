"""提供粒子滤波权重更新、有效粒子数和系统重采样。"""

import numpy as np
from scipy.special import logsumexp

from terrain_matching_foundation.utils.terrain import bilinear_sample


def update_log_weights(field, particles, log_weights, observation, sigma):
    """使用单条深度观测更新粒子对数权重。

    Args:
        field: 地形栅格。
        particles: 形状为 (P, 2) 的粒子位置。
        log_weights: 上一步归一化对数权重。
        observation: 当前深度观测。
        sigma: 正的深度观测标准差。

    Returns:
        tuple: 新的归一化对数权重和观测更新是否成功；无有效支持
            时保留原权重。
    """
    sampled = bilinear_sample(field, particles)
    valid = np.isfinite(sampled) & np.isfinite(log_weights)
    if not valid.any():
        return log_weights.copy(), False
    updated = np.full_like(log_weights, -np.inf)
    with np.errstate(over="ignore", invalid="ignore"):
        residual = (sampled[valid] - observation) / sigma
        updated[valid] = log_weights[valid] - 0.5 * residual**2
    normalization = logsumexp(updated)
    if not np.isfinite(normalization):
        return log_weights.copy(), False
    return updated - normalization, True


def effective_sample_size(weights):
    """计算归一化权重对应的有效粒子数。

    Args:
        weights: 一维归一化非负权重。

    Returns:
        float: 有效粒子数。
    """
    return float(1.0 / np.sum(weights**2))


def systematic_resample(weights, rng):
    """根据归一化权重执行系统重采样。

    Args:
        weights: 一维归一化非负权重。
        rng: 实例持有的 NumPy 随机数生成器。

    Returns:
        numpy.ndarray: 重采样粒子索引。
    """
    count = len(weights)
    positions = (rng.random() + np.arange(count)) / count
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, positions, side="right")
