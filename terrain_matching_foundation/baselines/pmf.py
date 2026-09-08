"""实现采用规则移动网格和单条深度观测的标准点质量滤波基线。"""

import logging

import numpy as np

from terrain_matching_foundation.baselines.base import BaseLocator
from terrain_matching_foundation.utils.particles import update_log_weights
from terrain_matching_foundation.utils.point_mass import (
    point_mass_grid,
    predict_log_mass,
)

logger = logging.getLogger(__name__)


class PMFLocator(BaseLocator):
    """采用 INS 位移预测和单点地形观测更新的二维点质量滤波器。"""

    def localize_trajectory(
        self, reference_trajectory, ins_trajectory, depth_observations
    ):
        """在固定间距的移动网格上逐点滤波并按统一窗口长度裁剪输出。

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
        if not starts:
            return corrected
        params = self._config["pmf"]
        grid = point_mass_grid(params["init_radius"], params["grid_step"])
        offsets = grid.reshape(-1, 2)
        log_mass = np.full(grid.shape[:2], -np.log(len(offsets)), dtype=np.float64)
        for index in starts:
            if index > 0:
                log_mass = predict_log_mass(
                    log_mass, params["process_std"], params["grid_step"]
                )
            # 直接锚定当前 INS，等价于逐步平移网格，避免累计浮点位移误差。
            points = ins[index] + offsets
            log_weights, updated = update_log_weights(
                self._field,
                points,
                log_mass.ravel(),
                depths[index],
                params["observation_std"],
            )
            if not updated:
                logger.warning(
                    "PMF step %d has no valid observation support; retaining prior",
                    index,
                )
            log_mass = log_weights.reshape(grid.shape[:2])
            weights = np.exp(log_weights)
            weights /= weights.sum()
            corrected[index] = weights @ points
        return corrected
