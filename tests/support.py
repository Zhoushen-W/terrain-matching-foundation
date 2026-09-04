"""提供测试专用地形、独立采样和临时配置工具。"""

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml
from scipy.ndimage import gaussian_filter


def synthetic_data(count=64, size=144, seed=101):
    """生成非对称地形及位于其内部的弯曲参考轨迹。

    Args:
        count: 轨迹点数。
        size: 方形地图边长。
        seed: 地形随机种子。

    Returns:
        tuple: 地形、参考轨迹和无噪声独立采样深度。
    """
    y, x = np.mgrid[:size, :size]
    rng = np.random.default_rng(seed)
    field = (
        6 * np.sin(x / 6.5)
        + 5 * np.cos(y / 7.5)
        + 3 * np.sin((x + 2 * y) / 9)
        + 0.004 * x * y
        + 12 * gaussian_filter(rng.normal(size=(size, size)), sigma=2)
    )
    time = np.linspace(0, 1, count)
    reference = np.column_stack((38 + 42 * time, 58 + 10 * np.sin(5 * time)))
    depths = np.array([scalar_sample(field, point) for point in reference])
    return field, reference, depths


def scalar_sample(field, point):
    """以独立标量公式采样，用于验证生产代码的向量化计算。

    Args:
        field: 地形数组。
        point: 位于有效地图范围内的 [x, y]。

    Returns:
        float: 双线性插值结果。
    """
    x, y = point
    left, top = math.floor(x), math.floor(y)
    right = min(left + 1, field.shape[1] - 1)
    bottom = min(top + 1, field.shape[0] - 1)
    horizontal, vertical = x - left, y - top
    upper = field[top, left] * (1 - horizontal) + field[top, right] * horizontal
    lower = field[bottom, left] * (1 - horizontal) + field[bottom, right] * horizontal
    return float(upper * (1 - vertical) + lower * vertical)


class ConfigTestCase(unittest.TestCase):
    """将测试过程文件集中保存在项目的 .codex_tmp 内。"""

    def setUp(self):
        """为当前测试创建自动清理的配置目录。

        Args:
            无。

        Returns:
            None: 注册临时目录清理操作。
        """
        root = Path(__file__).resolve().parents[1] / ".codex_tmp"
        root.mkdir(exist_ok=True)
        directory = tempfile.TemporaryDirectory(dir=root)
        self.addCleanup(directory.cleanup)
        self.config_directory = Path(directory.name)
        self.config_count = 0

    def config_path(self, overrides):
        """将局部配置写入当前测试的临时 YAML。

        Args:
            overrides: 要覆盖的配置字典。

        Returns:
            pathlib.Path: 临时配置路径。
        """
        self.config_count += 1
        path = self.config_directory / f"config_{self.config_count}.yaml"
        path.write_text(yaml.safe_dump(overrides), encoding="utf-8")
        return path
