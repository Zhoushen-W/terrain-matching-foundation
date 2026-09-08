"""实现每窗粗细两阶段 MAD 匹配与可选卡尔曼递推的 TERCOM。"""

import logging

import numpy as np

from terrain_matching_foundation.baselines.base import BaseLocator
from terrain_matching_foundation.utils.kalman import TransformKalman
from terrain_matching_foundation.utils.matching import mad_search

logger = logging.getLogger(__name__)


class TercomLocator(BaseLocator):
    """利用完整深度窗口估计二维平移的 TERCOM 定位器。"""

    def localize_trajectory(
        self, reference_trajectory, ins_trajectory, depth_observations
    ):
        """逐窗修正轨迹，每个完整窗口仅提交其起点。

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
        params = self._config["tercom"]
        kalman = TransformKalman(params["kalman"]) if params["use_kf"] else None
        state = np.zeros(2, dtype=np.float64)
        self.transform_ = state.copy()
        for start in starts:
            if kalman is not None and start > 0:
                state = kalman.predict()
            segment = ins[start : start + self._window_size]
            observed = depths[start : start + self._window_size]
            coarse, _ = mad_search(
                self._field,
                segment,
                observed,
                state,
                params["coarse_radius"],
                params["coarse_step"],
                self._batch_size,
            )
            if coarse is None:
                logger.warning(
                    "TERCOM window %d has no valid match; retaining prior", start
                )
            else:
                # 精搜索网格包含已验证有效的粗搜索中心。
                fine, _ = mad_search(
                    self._field,
                    segment,
                    observed,
                    coarse,
                    params["fine_radius"],
                    params["fine_step"],
                    self._batch_size,
                )
                state = kalman.update(fine) if kalman is not None else fine
            corrected[start] = ins[start] + state
        self.transform_ = state.copy()
        return corrected
