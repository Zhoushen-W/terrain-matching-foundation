"""提供等深线连续线段投影和二维刚体配准。"""

import contourpy
import numpy as np

from terrain_matching_foundation.utils.terrain import bilinear_sample


def rotation_matrix(angle):
    """构造二维正旋转矩阵。

    Args:
        angle: 弧度角度；在 y 向下的像素图中正角度表现为顺时针。

    Returns:
        numpy.ndarray: 形状为 (2, 2) 的旋转矩阵。
    """
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array([[cosine, -sine], [sine, cosine]])


def apply_transform(points, state, origin):
    """围绕固定原点应用二维旋转与平移。

    Args:
        points: 形状为 (..., 2) 的坐标。
        state: [dx, dy, theta] 变换状态。
        origin: 固定的旋转原点。

    Returns:
        numpy.ndarray: 变换后的坐标。
    """
    return (points - origin) @ rotation_matrix(state[2]).T + origin + state[:2]


def closest_on_segments(point, starts, ends):
    """穷举连续线段的最近投影，包含线段内部与端点。

    Args:
        point: 形状为 (2,) 的查询点。
        starts: 线段起点数组，形状为 (M, 2)。
        ends: 线段终点数组，形状为 (M, 2)。

    Returns:
        numpy.ndarray | None: 最近投影点，无线段时为 None。
    """
    if len(starts) == 0:
        return None
    direction = ends - starts
    lengths = np.sum(direction * direction, axis=1)
    fraction = np.divide(
        np.sum((point - starts) * direction, axis=1),
        lengths,
        out=np.zeros_like(lengths),
        where=lengths > 0,
    )
    projections = starts + np.clip(fraction, 0, 1)[:, None] * direction
    distances = np.sum((projections - point) ** 2, axis=1)
    return projections[np.argmin(distances)].copy()


def fit_rigid_transform(source, target):
    """求解禁止反射的二维最小二乘刚体变换。

    Args:
        source: 非空的原始对应点，形状为 (M, 2)。
        target: 目标对应点，形状与 source 相同。

    Returns:
        tuple: 旋转矩阵、平移向量及旋转是否可确定的标记；退化时
            返回单位旋转和质心平移。
    """
    source_mean = np.mean(source, axis=0)
    target_mean = np.mean(target, axis=0)
    centered_source = source - source_mean
    centered_target = target - target_mean
    dot = np.sum(centered_source * centered_target)
    cross = np.sum(
        centered_source[:, 0] * centered_target[:, 1]
        - centered_source[:, 1] * centered_target[:, 0]
    )
    scale = np.sqrt(np.sum(centered_source**2) * np.sum(centered_target**2))
    observable = bool(scale > 0 and np.hypot(dot, cross) > 1e-12 * scale)
    angle = np.arctan2(cross, dot) if observable else 0.0
    rotation = rotation_matrix(angle)
    translation = target_mean - rotation @ source_mean
    return rotation, translation, observable


class TerrainContours:
    """通过全图等深线提供 ICCP 的几何对应关系。"""

    def __init__(self, field):
        """建立采用像素坐标和完整有效网格单元的等深线生成器。

        Args:
            field: 经过校验的地形栅格。

        Returns:
            None: 初始化等深线生成器。
        """
        self.field = field
        self.generator = contourpy.contour_generator(
            z=np.ma.masked_invalid(field), corner_mask=False, line_type="Separate"
        )

    def segments(self, level):
        """提取指定深度在整张地图上的所有连续线段。

        Args:
            level: 有限深度值。

        Returns:
            tuple: 形状均为 (M, 2) 的线段起点及终点。
        """
        lines = self.generator.lines(float(level))
        lines = [line for line in lines if len(line) >= 2]
        if not lines:
            return np.empty((0, 2)), np.empty((0, 2))
        return (
            np.concatenate([line[:-1] for line in lines]),
            np.concatenate([line[1:] for line in lines]),
        )

    def correspondences(self, points, levels, segments):
        """寻找窗口各点在其观测等深线上的最近对应点。

        Args:
            points: 当前变换后的窗口轨迹。
            levels: 与轨迹对应的深度观测。
            segments: 各观测对应的完整线段列表。

        Returns:
            tuple: 对应点数组及有效对应的布尔掩码。
        """
        targets = np.full_like(points, np.nan)
        sampled = bilinear_sample(self.field, points)
        for index, (point, level, lines) in enumerate(zip(points, levels, segments)):
            # 已位于平坦等深区域的点，其最近对应点就是自身。
            if sampled[index] == level:
                targets[index] = point
            else:
                closest = closest_on_segments(point, *lines)
                if closest is not None:
                    targets[index] = closest
        return targets, np.isfinite(targets).all(axis=1)
