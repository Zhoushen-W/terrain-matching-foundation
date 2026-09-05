"""实现全图等深线刚体配准及可选卡尔曼递推的 ICCP。"""

import logging
from collections import deque

import numpy as np

from baselines.base import BaseLocator
from utils.geometry import TerrainContours, apply_transform, fit_rigid_transform
from utils.kalman import TransformKalman, wrap_angle
from utils.matching import mad_search
from utils.terrain import bilinear_sample

logger = logging.getLogger(__name__)


class IccpLocator(BaseLocator):
    """以一次 TERCOM 粗匹配初始化的二维等深线配准定位器。"""

    def _refine(self, segment, observed, initial, origin, contours, segments):
        """交替计算最近等深线对应点和二维刚体变换。

        Args:
            segment: 原始 INS 窗口。
            observed: 窗口深度观测。
            initial: 初始 [dx, dy, theta]。
            origin: 本次轨迹调用的固定旋转原点。
            contours: 等深线查询工具。
            segments: 与窗口观测逐一对应的线段集合。

        Returns:
            tuple: 最终变换及角度是否有约束；无有效解时变换为 None。
        """
        params = self._config["iccp"]
        state = initial.copy()
        matched = False
        angle_observed = False
        for _ in range(params["max_iterations"]):
            points = apply_transform(segment, state, origin)
            targets, valid = contours.correspondences(points, observed, segments)
            if not valid.any():
                break
            rotation, translation, observable = fit_rigid_transform(
                points[valid], targets[valid]
            )
            angle_step = np.arctan2(rotation[1, 0], rotation[0, 0])
            next_state = np.empty(3, dtype=np.float64)
            next_state[:2] = rotation @ (origin + state[:2]) + translation - origin
            next_state[2] = wrap_angle(state[2] + angle_step)
            movement = np.linalg.norm(next_state[:2] - state[:2])
            state = next_state
            matched = True
            angle_observed = angle_observed or observable
            if (
                movement <= params["translation_tolerance"]
                and abs(angle_step) <= params["angle_tolerance"]
            ):
                break
        points = apply_transform(segment, state, origin)
        if not matched or not np.isfinite(bilinear_sample(self._field, points)).all():
            return None, False
        return state, angle_observed

    def localize_trajectory(
        self, reference_trajectory, ins_trajectory, depth_observations
    ):
        """逐窗估计平移和旋转，并只返回窗口起点的修正位置。

        Args:
            reference_trajectory: 形状为 (N, 2) 的参考轨迹，仅校验输入。
            ins_trajectory: 形状为 (N, 2) 的原始 INS 像素轨迹。
            depth_observations: 形状为 (N,) 的深度观测。

        Returns:
            numpy.ndarray: 形状为 (max(N-L+1, 0), 2) 的 float64 修正轨迹。

        Raises:
            ValueError: 输入形状不合法。
        """
        ins, depths, starts, corrected = self._prepare(
            reference_trajectory, ins_trajectory, depth_observations
        )
        params = self._config["iccp"]
        state = np.zeros(3, dtype=np.float64)
        self.transform_ = state.copy()
        if not starts:
            return corrected
        origin = ins[0].copy()
        kalman = (
            TransformKalman(params["kalman"], with_angle=True)
            if params["use_kf"]
            else None
        )
        contours = TerrainContours(self._field)
        # 仅保留当前窗口的等深线几何，不对深度进行量化。
        segments = deque(
            contours.segments(level) for level in depths[: self._window_size]
        )
        for start in starts:
            if start > 0:
                segments.popleft()
                segments.append(
                    contours.segments(depths[start + self._window_size - 1])
                )
                if kalman is not None:
                    state = kalman.predict()
            initial = state.copy()
            segment = ins[start : start + self._window_size]
            observed = depths[start : start + self._window_size]
            coarse = None
            if start == 0:
                coarse, _ = mad_search(
                    self._field,
                    segment,
                    observed,
                    initial[:2],
                    params["init_radius"],
                    params["init_step"],
                    self._batch_size,
                )
                if coarse is not None:
                    initial[:2] = coarse
                else:
                    logger.warning("ICCP initial TERCOM search has no valid match")
            measurement, angle_observed = self._refine(
                segment, observed, initial, origin, contours, segments
            )
            if measurement is None:
                logger.warning(
                    "ICCP window %d has no valid contour fit; retaining prior", start
                )
                if coarse is not None:
                    measurement = initial
            if measurement is not None:
                state = (
                    kalman.update(measurement, observe_angle=angle_observed)
                    if kalman is not None
                    else measurement
                )
            corrected[start] = apply_transform(ins[start], state, origin)
        self.transform_ = state.copy()
        return corrected
