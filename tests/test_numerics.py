"""独立校验插值、MAD 搜索、等深线几何、KF 和粒子滤波数值。"""

import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from tests.support import scalar_sample, synthetic_data
from utils.geometry import TerrainContours, closest_on_segments, fit_rigid_transform
from utils.kalman import TransformKalman, wrap_angle
from utils.matching import mad_search, search_axis
from utils.particles import (
    effective_sample_size,
    systematic_resample,
    update_log_weights,
)
from utils.terrain import bilinear_sample, prepare_terrain


class NumericalTests(unittest.TestCase):
    """用解析结果及独立实现检验公共数值组件。"""

    def test_sampling_matches_independent_scalar(self):
        """验证双线性采样与独立标量公式一致。

        Args:
            无。

        Returns:
            None: 断言采样结果一致。
        """
        field, points, _ = synthetic_data(count=23)
        expected = [scalar_sample(field, point) for point in points]
        assert_allclose(bilinear_sample(field, points), expected, atol=1e-12)
        assert_allclose(bilinear_sample(field, points.reshape(1, 23, 2)), [expected])

    def test_sampling_edges_and_nan_weights(self):
        """验证整数边界和 NaN 邻点的有效权重语义。

        Args:
            无。

        Returns:
            None: 断言边界和无效区域处理正确。
        """
        field = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        points = np.array([[0, 0], [2, 2], [1, 1], [0.5, 0], [-0.01, 1], [2.01, 1]])
        actual = bilinear_sample(field, points)
        assert_array_equal(actual[:3], [1, 9, 5])
        self.assertTrue(np.isnan(actual[3:]).all())
        prepared = prepare_terrain(field)
        self.assertFalse(prepared.flags.writeable)
        self.assertFalse(np.shares_memory(prepared, field))

    def test_search_matches_exhaustive_and_batch_sizes(self):
        """验证分块 MAD 与独立穷举及所有分块大小一致。

        Args:
            无。

        Returns:
            None: 断言最优偏移及代价一致。
        """
        field, truth, depths = synthetic_data(count=17)
        ins = truth + [1.5, -0.5]
        center = np.zeros(2)
        entries = []
        for dx in (-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0):
            for dy in (-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0):
                samples = [scalar_sample(field, p + [dx, dy]) for p in ins]
                cost = np.mean(np.abs(np.array(samples) - depths))
                entries.append((cost, dx * dx + dy * dy, dx, dy))
        best = min(entries)
        for batch in (1, 7, 1024):
            shift, cost = mad_search(field, ins, depths, center, 2, 0.5, batch)
            assert_allclose(shift, best[2:], atol=1e-12)
            self.assertAlmostEqual(cost, best[0], places=12)

    def test_search_grid_ties_and_failure(self):
        """验证非整除网格、平坦图先验及无有效候选。

        Args:
            无。

        Returns:
            None: 断言同分规则和失败返回正确。
        """
        assert_allclose(search_axis(1, 0.6), [-1, -0.6, 0, 0.6, 1])
        assert_array_equal(search_axis(0, 1), [0])
        field = np.ones((10, 10))
        points = np.array([[4.0, 4.0], [5.0, 5.0]])
        for batch in (1, 3, 1024):
            shift, _ = mad_search(field, points, np.ones(2), [0.25, 0.5], 2, 1, batch)
            assert_allclose(shift, [0.25, 0.5])
        shift, cost = mad_search(field, points + 100, np.ones(2), [0, 0], 1, 1, 10)
        self.assertIsNone(shift)
        self.assertTrue(np.isinf(cost))

    def test_segment_projection_matches_scalar_geometry(self):
        """验证连续线段投影对照独立逐线段计算。

        Args:
            无。

        Returns:
            None: 断言投影与暴力结果一致。
        """
        rng = np.random.default_rng(42)
        starts, ends = rng.normal(size=(2, 50, 2))
        ends[0] = starts[0]
        for point in rng.normal(size=(9, 2)):
            candidates = []
            for first, last in zip(starts, ends):
                vector = last - first
                denominator = float(np.dot(vector, vector))
                fraction = (
                    float(np.dot(point - first, vector)) / denominator
                    if denominator
                    else 0
                )
                candidates.append(first + max(0, min(1, fraction)) * vector)
            expected = min(candidates, key=lambda p: np.linalg.norm(p - point))
            assert_allclose(closest_on_segments(point, starts, ends), expected)
        assert_allclose(
            closest_on_segments(
                np.array([3.0, 2.0]), np.array([[0.0, 0.0]]), np.array([[10.0, 0.0]])
            ),
            [3, 0],
        )

    def test_contours_use_pixel_axes_and_mask(self):
        """验证等深线坐标方向、连续投影及无效区域。

        Args:
            无。

        Returns:
            None: 断言解析平面上的等深线位置正确。
        """
        y, x = np.mgrid[:12, :12]
        field = (x + 2 * y).astype(float)
        contours = TerrainContours(field)
        lines = contours.segments(8)
        closest = closest_on_segments(np.array([4.0, 4.0]), *lines)
        assert_allclose(closest, [3.2, 2.4], atol=1e-12)
        self.assertEqual(len(contours.segments(-1)[0]), 0)
        field[2:6, 2:6] = np.nan
        masked = TerrainContours(field).segments(8)
        midpoints = (masked[0] + masked[1]) / 2
        self.assertTrue(np.isfinite(bilinear_sample(field, midpoints)).all())

    def test_rigid_transform_and_degeneracy(self):
        """验证刚体求解恢复旋转平移，并处理反射和退化点集。

        Args:
            无。

        Returns:
            None: 断言恢复结果及旋转行列式正确。
        """
        source = np.array([[1.0, 1.0], [4.0, 2.0], [2.0, 5.0], [8.0, -1.0]])
        expected_rotation = np.array([[0.8, -0.6], [0.6, 0.8]])
        target = source @ expected_rotation.T + [2, -3]
        rotation, translation, observable = fit_rigid_transform(source, target)
        assert_allclose(rotation, expected_rotation, atol=1e-12)
        assert_allclose(translation, [2, -3], atol=1e-12)
        self.assertTrue(observable)
        reflected = source * [-1, 1]
        rotation, _, _ = fit_rigid_transform(source, reflected)
        self.assertAlmostEqual(np.linalg.det(rotation), 1)
        rotation, translation, observable = fit_rigid_transform(source[:1], target[:1])
        assert_array_equal(rotation, np.eye(2))
        assert_allclose(translation, target[0] - source[0])
        self.assertFalse(observable)

    def test_kalman_matches_analytic_update(self):
        """验证平移 KF 对照解析标量更新并保持正定。

        Args:
            无。

        Returns:
            None: 断言状态与协方差正确。
        """
        kalman = TransformKalman(
            {"initial_std": 2, "process_std": 1, "measurement_std": 3}
        )
        kalman.predict()
        assert_allclose(kalman.update([7.0, -7.0]), [2.5, -2.5])
        assert_allclose(kalman.covariance, np.eye(2) * 45 / 14)
        for _ in range(100):
            kalman.predict()
            kalman.update([1.0, 2.0])
        self.assertTrue((np.linalg.eigvalsh(kalman.covariance) > 0).all())

    def test_kalman_angle_wrap_and_partial_observation(self):
        """验证跨越正负 π 的创新和缺少旋转观测时的状态保持。

        Args:
            无。

        Returns:
            None: 断言角度沿短弧更新且不会虚增旋转置信度。
        """
        config = {
            "initial_std": 1,
            "process_std": 0,
            "measurement_std": 1,
            "initial_angle_std_deg": 1,
            "process_angle_std_deg": 0,
            "measurement_angle_std_deg": 1,
        }
        kalman = TransformKalman(config, with_angle=True)
        kalman.state[2] = np.deg2rad(179)
        posterior = kalman.update([0.0, 0.0, np.deg2rad(-179)])
        self.assertLess(abs(wrap_angle(posterior[2] - np.pi)), 1e-10)
        variance = kalman.covariance[2, 2]
        kalman.update([1.0, 1.0, 0.0], observe_angle=False)
        self.assertAlmostEqual(kalman.covariance[2, 2], variance)
        self.assertLess(abs(wrap_angle(kalman.state[2] - np.pi)), 1e-10)

    def test_particle_weights_and_failure(self):
        """验证单观测贝叶斯权重及全部越界时的先验保持。

        Args:
            无。

        Returns:
            None: 断言权重符合解析结果且失败不重置分布。
        """
        y, x = np.mgrid[:10, :10]
        field = x + 2 * y
        particles = np.array([[2.0, 3.0], [4.0, 1.0], [100.0, 100.0]])
        prior = np.log([0.2, 0.3, 0.5])
        updated, success = update_log_weights(field, particles, prior, 6, 2)
        expected = np.array([0.2 * np.exp(-0.5), 0.3, 0])
        assert_allclose(np.exp(updated), expected / expected.sum())
        self.assertTrue(success)
        retained, success = update_log_weights(field, particles + 100, prior, 6, 2)
        assert_array_equal(retained, prior)
        self.assertFalse(success)
        retained, success = update_log_weights(field, particles, prior, 1e308, 1e-308)
        self.assertFalse(success)
        assert_array_equal(retained, prior)

    def test_systematic_resampling(self):
        """验证重采样等价于独立逆累积分布且排除零权重。

        Args:
            无。

        Returns:
            None: 断言重采样索引及有效粒子数正确。
        """
        weights = np.array([0.0, 0.2, 0.3, 0.5])
        expected_rng = np.random.default_rng(9)
        positions = (expected_rng.random() + np.arange(4)) / 4
        expected = [
            next(j for j in range(4) if p < sum(weights[: j + 1])) for p in positions
        ]
        indices = systematic_resample(weights, np.random.default_rng(9))
        assert_array_equal(indices, expected)
        self.assertNotIn(0, indices)
        self.assertAlmostEqual(effective_sample_size(np.ones(10) / 10), 10)
