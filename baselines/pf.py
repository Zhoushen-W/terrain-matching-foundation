"""实现采用单条深度观测更新的标准逐点粒子滤波基线。"""

import logging

import numpy as np

from baselines.base import BaseLocator
from utils.particles import (
    effective_sample_size,
    systematic_resample,
    update_log_weights,
)

logger = logging.getLogger(__name__)


class PFLocator(BaseLocator):
    """采用 INS 位移预测和单点地形观测更新的二维粒子滤波器。"""

    def localize_trajectory(
        self, reference_trajectory, ins_trajectory, depth_observations
    ):
        """逐点滤波并按照统一窗口长度决定停止位置。

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
        params = self._config["pf"]
        rng = np.random.default_rng(params["seed"])
        count = params["num_particles"]
        particles = ins[0] + rng.uniform(
            -params["init_radius"], params["init_radius"], size=(count, 2)
        )
        log_weights = np.full(count, -np.log(count), dtype=np.float64)
        for index in starts:
            if index > 0:
                particles += ins[index] - ins[index - 1]
                if params["process_std"] > 0:
                    particles += rng.normal(0, params["process_std"], particles.shape)
            log_weights, updated = update_log_weights(
                self._field,
                particles,
                log_weights,
                depths[index],
                params["observation_std"],
            )
            if not updated:
                logger.warning(
                    "PF step %d has no valid observation support; retaining prior",
                    index,
                )
            weights = np.exp(log_weights)
            weights /= weights.sum()
            corrected[index] = weights @ particles
            if (
                updated
                and effective_sample_size(weights)
                < params["resample_threshold"] * count
            ):
                indices = systematic_resample(weights, rng)
                particles = particles[indices]
                log_weights.fill(-np.log(count))
        return corrected
