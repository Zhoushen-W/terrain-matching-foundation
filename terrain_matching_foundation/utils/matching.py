"""提供 TERCOM 使用的分块 MAD 平移搜索。"""

import numpy as np

from terrain_matching_foundation.utils.terrain import bilinear_sample


def search_axis(radius, step):
    """建立包含中心和边界的对称搜索偏移。

    Args:
        radius: 非负搜索半径。
        step: 正的网格间距。

    Returns:
        numpy.ndarray: 升序的一维偏移，边界剩余间距可能小于 step。
    """
    count = int(np.floor(radius / step))
    axis = np.arange(-count, count + 1, dtype=np.float64) * step
    if count * step < radius:
        axis = np.concatenate(([-radius], axis, [radius]))
    return axis


def mad_search(field, ins_segment, measured, center, radius, step, batch_size):
    """以整窗 MAD 搜索二维平移，并按先验距离确定同分候选。

    Args:
        field: 地形栅格。
        ins_segment: 原始 INS 窗口，形状为 (L, 2)。
        measured: 窗口深度观测，形状为 (L,)。
        center: 相对于原始 INS 的搜索中心平移。
        radius: 搜索半径，单位为像素。
        step: 搜索间距，单位为像素。
        batch_size: 每次计算的候选数。

    Returns:
        tuple: 最优平移和 MAD；无有效候选时为 (None, inf)。
    """
    axis = search_axis(radius, step)
    dx, dy = np.meshgrid(axis, axis)
    offsets = np.column_stack((dx.ravel(), dy.ravel()))
    candidates = offsets + center
    best_key = (np.inf, np.inf, np.inf, np.inf)
    best = None
    for start in range(0, len(candidates), batch_size):
        shifts = candidates[start : start + batch_size]
        samples = bilinear_sample(field, ins_segment[None, :, :] + shifts[:, None, :])
        valid = np.isfinite(samples).all(axis=1)
        if not valid.any():
            continue
        shifts = shifts[valid]
        costs = np.mean(np.abs(samples[valid] - measured), axis=1)
        distances = np.sum((shifts - center) ** 2, axis=1)
        order = np.lexsort((shifts[:, 1], shifts[:, 0], distances, costs))
        index = order[0]
        key = (costs[index], distances[index], shifts[index, 0], shifts[index, 1])
        if key < best_key:
            best_key = key
            best = shifts[index].copy()
    return best, float(best_key[0])
