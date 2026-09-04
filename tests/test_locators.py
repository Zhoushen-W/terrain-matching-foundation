"""验证三种定位器的轨迹精度、输出语义和递推边界。"""

from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from baselines import IccpLocator, PFLocator, TercomLocator
from tests.evaluate_accuracy import accuracy_cases, error_metrics, evaluation_config
from tests.support import ConfigTestCase, synthetic_data
from utils.geometry import rotation_matrix
from utils.matching import mad_search
from utils.particles import update_log_weights

LOCATORS = (TercomLocator, IccpLocator, PFLocator)


class LocatorTests(ConfigTestCase):
    """验证用户可见接口及完整定位过程。"""

    def small_config(self, window_size=8):
        """创建适用于接口测试的小规模配置。

        Args:
            window_size: 窗口长度。

        Returns:
            dict: 小范围搜索和粒子参数的覆盖配置。
        """
        return {
            "common": {"window_size": window_size},
            "tercom": {"coarse_radius": 4, "coarse_step": 1},
            "iccp": {"init_radius": 4, "init_step": 1},
            "pf": {"num_particles": 512, "init_radius": 4, "process_std": 0.1},
        }

    def test_lengths_and_output_indices(self):
        """验证完整窗口、短轨迹、空轨迹与逐点窗口的输出索引。

        Args:
            无。

        Returns:
            None: 断言输出只包含正确的轨迹前缀。
        """
        field = np.ones((20, 20))
        for length, window_size in ((10, 4), (4, 4), (3, 4), (0, 4), (5, 1)):
            config = self.small_config(window_size)
            config["pf"].update(init_radius=0, process_std=0)
            path = self.config_path(config)
            ins = np.column_stack((np.arange(length) * 0.3 + 5, np.full(length, 8)))
            for locator_type in LOCATORS:
                with self.subTest(
                    length=length, window=window_size, locator=locator_type
                ):
                    actual = locator_type(field, path).localize_trajectory(
                        ins, ins, np.ones(length)
                    )
                    expected = ins[: max(0, length - window_size + 1)]
                    self.assertEqual(actual.shape, expected.shape)
                    self.assertEqual(actual.dtype, np.float64)
                    assert_allclose(actual, expected, atol=1e-12)
                    self.assertFalse(np.shares_memory(actual, ins))

    def test_reference_isolation_input_immutability_and_reset(self):
        """验证参考轨迹隔离、输入保持及每次调用状态重置。

        Args:
            无。

        Returns:
            None: 断言相同输入和随机种子的定位完全可复现。
        """
        field, reference, depths = synthetic_data(count=16)
        ins = reference + [2, -1]
        originals = [array.copy() for array in (field, reference, ins, depths)]
        path = self.config_path(self.small_config())
        for locator_type in LOCATORS:
            locator = locator_type(field, path)
            first = locator.localize_trajectory(reference, ins, depths)
            altered_reference = reference * -100 + 123456
            altered = locator.localize_trajectory(altered_reference, ins, depths)
            locator.localize_trajectory(reference, ins + [1, 0], depths + 0.1)
            repeated = locator.localize_trajectory(reference, ins, depths)
            assert_array_equal(first, altered)
            assert_array_equal(first, repeated)
        for original, current in zip(originals, (field, reference, ins, depths)):
            assert_array_equal(original, current)

    def test_validation_and_invalid_config(self):
        """验证非法数据、非有限输入和不支持的配置被明确拒绝。

        Args:
            无。

        Returns:
            None: 断言配置和输入错误均触发 ValueError。
        """
        field, reference, depths = synthetic_data(count=8)
        path = self.config_path(self.small_config())
        cases = (
            (reference[:, :1], reference, depths),
            (reference, reference[:-1], depths),
            (reference, reference, depths[:, None]),
            (reference, reference.astype(complex), depths),
            (reference, reference + np.nan, depths),
            (reference, reference, depths + np.inf),
        )
        for locator_type in LOCATORS:
            locator = locator_type(field, path)
            for arrays in cases:
                with self.assertRaises(ValueError):
                    locator.localize_trajectory(*arrays)
            for invalid_map in (np.ones((1, 8)), field + np.inf, field.astype(complex)):
                with self.assertRaises(ValueError):
                    locator_type(invalid_map, path)
        configs = (
            {"common": {"stride": 2}},
            {"common": {"window_size": 0}},
            {"pf": {"num_particles": 3.2}},
            {"pf": {"seed": -1}},
            {"pf": {"observation_std": 0}},
            {"pf": {"resample_threshold": 1.1}},
            {"tercom": {"use_kf": "false"}},
            {"common": {"misspelled": 1}},
            {"iccp": {"init_step": float("nan")}},
            {"common": {"window_size": True}},
        )
        for config in configs:
            with self.assertRaises(ValueError):
                TercomLocator(field, self.config_path(config))

    def test_tercom_two_stages_per_window(self):
        """验证每个有效窗口都有粗精搜索并使用最终偏移递推。

        Args:
            无。

        Returns:
            None: 断言搜索阶段数量与中心正确。
        """
        field, reference, depths = synthetic_data(count=12)
        ins = reference + [2, -1]
        path = self.config_path(self.small_config())
        with patch("baselines.tercom.mad_search", wraps=mad_search) as search:
            corrected = TercomLocator(field, path).localize_trajectory(
                reference, ins, depths
            )
        self.assertEqual(search.call_count, 2 * len(corrected))
        for index in range(1, len(corrected)):
            center = search.call_args_list[2 * index].args[3]
            assert_allclose(center, corrected[index - 1] - ins[index - 1])
        assert_allclose(corrected, reference[: len(corrected)], atol=1e-10)

    def test_tercom_subpixel_precision_and_kf(self):
        """验证无噪声亚像素平移恢复及两种 KF 模式。

        Args:
            无。

        Returns:
            None: 断言修正精度达到亚像素级。
        """
        field, reference, depths = synthetic_data(count=40)
        ins = reference + [2.75, -1.25]
        for enabled in (False, True):
            config = self.small_config(24)
            config["tercom"]["use_kf"] = enabled
            corrected = TercomLocator(
                field, self.config_path(config)
            ).localize_trajectory(reference, ins, depths)
            error = np.linalg.norm(corrected - reference[: len(corrected)], axis=1)
            self.assertLess(np.max(error), 0.3)

    def test_iccp_rotation_and_one_initialization(self):
        """验证 ICCP 恢复已知旋转且每次调用只执行一次粗初始化。

        Args:
            无。

        Returns:
            None: 断言定位与旋转精度以及初始化次数。
        """
        field, reference, depths = synthetic_data(count=48)
        angle = np.deg2rad(1.5)
        ins = (
            (reference - reference[0]) @ rotation_matrix(-angle).T
            + reference[0]
            + [2.5, -3]
        )
        for enabled in (False, True):
            config = self.small_config(32)
            config["iccp"]["use_kf"] = enabled
            locator = IccpLocator(field, self.config_path(config))
            with patch("baselines.iccp.mad_search", wraps=mad_search) as search:
                corrected = locator.localize_trajectory(reference, ins, depths)
                repeated = locator.localize_trajectory(reference, ins, depths)
            self.assertEqual(search.call_count, 2)
            assert_array_equal(corrected, repeated)
            error = np.linalg.norm(corrected - reference[: len(corrected)], axis=1)
            self.assertLess(np.sqrt(np.mean(error**2)), 0.5)
            self.assertLess(abs(np.rad2deg(locator.transform_[2] - angle)), 0.5)

    def test_pf_uses_only_single_observations_and_stops(self):
        """验证 PF 每步仅使用当前观测，并不读取停止位置之后的轨迹。

        Args:
            无。

        Returns:
            None: 断言观测调用序列与未来数据隔离。
        """
        field, reference, depths = synthetic_data(count=16)
        ins = reference + [2, -1]
        locator = PFLocator(field, self.config_path(self.small_config()))
        with patch(
            "baselines.pf.update_log_weights", wraps=update_log_weights
        ) as update:
            first = locator.localize_trajectory(reference, ins, depths)
        self.assertEqual(update.call_count, len(first))
        used_depths = [call.args[3] for call in update.call_args_list]
        assert_array_equal(used_depths, depths[: len(first)])
        alternate_depths = depths.copy()
        alternate_depths[len(first) :] += 1000
        alternate_ins = ins.copy()
        alternate_ins[len(first) :] += 1000
        second = locator.localize_trajectory(reference, alternate_ins, alternate_depths)
        assert_array_equal(first, second)

    def test_pf_prediction_and_estimate_before_resampling(self):
        """验证无过程噪声的 INS 位移传播及输出早于重采样。

        Args:
            无。

        Returns:
            None: 断言预测位移和估计顺序正确。
        """
        field, reference, depths = synthetic_data(count=10)
        config = self.small_config(4)
        config["pf"].update(process_std=0, resample_threshold=1)
        path = self.config_path(config)
        with patch(
            "baselines.pf.update_log_weights", wraps=update_log_weights
        ) as update:
            actual = PFLocator(field, path).localize_trajectory(
                reference, reference, depths
            )
        first_particles = update.call_args_list[0].args[1]
        rng = np.random.default_rng(0)
        initial = reference[0] + rng.uniform(-4, 4, size=(512, 2))
        # mock references may share later in-place predictions, so use a fresh cloud.
        logs, _ = update_log_weights(
            field, initial, np.full(512, -np.log(512)), depths[0], 1
        )
        expected = np.exp(logs) @ initial
        assert_allclose(actual[0], expected, atol=1e-12)
        self.assertEqual(first_particles.shape, (512, 2))
        config["pf"].update(init_radius=0, process_std=0)
        result = PFLocator(field, self.config_path(config)).localize_trajectory(
            reference, reference, depths
        )
        assert_allclose(result, reference[: len(result)], atol=1e-11)

    def test_missing_map_support_retains_prior(self):
        """验证整图无效时沿用先验且不裁剪越界轨迹。

        Args:
            无。

        Returns:
            None: 断言失败日志和先验位置正确。
        """
        field = np.full((12, 12), np.nan)
        ins = np.column_stack((np.arange(8) + 100.0, np.arange(8) + 100.0))
        config = self.small_config(4)
        config["pf"].update(init_radius=0, process_std=0)
        for enabled in (False, True):
            config["tercom"]["use_kf"] = enabled
            config["iccp"]["use_kf"] = enabled
            path = self.config_path(config)
            for locator_type in LOCATORS:
                with self.assertLogs("baselines", level="WARNING"):
                    result = locator_type(field, path).localize_trajectory(
                        ins, ins, np.ones(8)
                    )
                assert_allclose(result, ins[:5], atol=1e-10)

    def test_drift_noise_and_pf_seed_stability(self):
        """验证独立地形上的漂移校正、噪声变化和 PF 多种子精度。

        Args:
            无。

        Returns:
            None: 断言低噪声校正有效及高噪声结果保持有限可复现。
        """
        low_case, high_case = accuracy_cases()[-2:]
        particle_errors = []
        for case in (low_case, high_case):
            config = evaluation_config(case)
            count = len(case["ins"]) - config["common"]["window_size"] + 1
            initial = error_metrics(case["ins"][:count], case["reference"])["rmse_px"]
            errors = []
            for seed in range(5):
                config["pf"]["seed"] = seed
                result = PFLocator(
                    case["field"], self.config_path(config)
                ).localize_trajectory(case["reference"], case["ins"], case["depths"])
                errors.append(error_metrics(result, case["reference"])["rmse_px"])
            self.assertLess(max(errors), initial)
            self.assertLess(np.std(errors), 0.2)
            particle_errors.append(float(np.median(errors)))
            for locator_type, section in (
                (TercomLocator, "tercom"),
                (IccpLocator, "iccp"),
            ):
                for enabled in (False, True):
                    config[section]["use_kf"] = enabled
                    result = locator_type(
                        case["field"], self.config_path(config)
                    ).localize_trajectory(
                        case["reference"], case["ins"], case["depths"]
                    )
                    self.assertTrue(np.isfinite(result).all())
                    error = error_metrics(result, case["reference"])["rmse_px"]
                    if case is low_case or locator_type is TercomLocator or enabled:
                        self.assertLess(error, initial)
        self.assertGreater(particle_errors[1], particle_errors[0])
