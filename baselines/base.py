"""统一基线配置、地形所有权和输入准备逻辑。"""

from pathlib import Path

import numpy as np

from utils.config import load_config
from utils.inputs import validate_trajectories, window_starts
from utils.terrain import prepare_terrain


class BaseLocator:
    """供三个定位器复用的轻量公共基类。"""

    def __init__(self, terrain_map, config_path=None):
        """加载配置并持有独立的只读地图。

        Args:
            terrain_map: 形状为 (H, W) 的地形 NumPy 数组。
            config_path: 可选的完整或局部 YAML 配置路径。

        Returns:
            None: 初始化定位器。

        Raises:
            ValueError: 配置或地形不合法。
            OSError: 配置文件无法读取。
        """
        self._config = load_config(Path(__file__).with_name("config.yaml"), config_path)
        self._field = prepare_terrain(terrain_map)
        self._window_size = self._config["common"]["window_size"]
        self._batch_size = self._config["common"]["candidate_batch_size"]

    def _prepare(self, reference_trajectory, ins_trajectory, depth_observations):
        """校验输入并分配经过统一尾部裁剪的输出。

        Args:
            reference_trajectory: 仅用于外部评估的参考轨迹。
            ins_trajectory: 原始 INS 轨迹。
            depth_observations: 实际输入的深度观测序列。

        Returns:
            tuple: INS、深度、窗口起点以及独立输出数组。

        Raises:
            ValueError: 输入形状或数值不合法。
        """
        ins, depths = validate_trajectories(
            reference_trajectory, ins_trajectory, depth_observations
        )
        starts = window_starts(len(ins), self._window_size)
        corrected = np.empty((len(starts), 2), dtype=np.float64)
        return ins, depths, starts, corrected
