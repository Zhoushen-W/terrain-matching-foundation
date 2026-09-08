"""提供平移及旋转变换的随机游走卡尔曼递推。"""

import numpy as np


def wrap_angle(angle):
    """将角度归一化到 [-π, π)。

    Args:
        angle: 弧度角度标量或数组。

    Returns:
        float | numpy.ndarray: 归一化角度。
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi


class TransformKalman:
    """以平移和可选旋转角为状态的随机游走滤波器。"""

    def __init__(self, config, with_angle=False):
        """按标准差参数建立零均值先验。

        Args:
            config: 包含初始、过程及观测标准差的配置。
            with_angle: 是否加入第三个旋转角状态。

        Returns:
            None: 初始化滤波器。
        """
        initial = [config["initial_std"]] * 2
        process = [config["process_std"]] * 2
        measurement = [config["measurement_std"]] * 2
        if with_angle:
            initial.append(np.deg2rad(config["initial_angle_std_deg"]))
            process.append(np.deg2rad(config["process_angle_std_deg"]))
            measurement.append(np.deg2rad(config["measurement_angle_std_deg"]))
        self.state = np.zeros(len(initial), dtype=np.float64)
        self.covariance = np.diag(np.square(initial))
        self.process_covariance = np.diag(np.square(process))
        self.measurement_covariance = np.diag(np.square(measurement))
        self.with_angle = with_angle

    def predict(self):
        """进行一个窗口步长的随机游走预测。

        Args:
            无。

        Returns:
            numpy.ndarray: 预测状态副本。
        """
        self.covariance += self.process_covariance
        return self.state.copy()

    def update(self, measurement, observe_angle=True):
        """融合匹配结果并用 Joseph 形式更新协方差。

        Args:
            measurement: 与完整状态同维的匹配变换。
            observe_angle: 三维状态时是否具有有效角度观测。

        Returns:
            numpy.ndarray: 后验状态副本。
        """
        dimensions = len(self.state)
        count = 2 if self.with_angle and not observe_angle else dimensions
        observation = np.eye(dimensions)[:count]
        noise = self.measurement_covariance[:count, :count]
        innovation = np.asarray(measurement)[:count] - observation @ self.state
        if self.with_angle and observe_angle:
            innovation[2] = wrap_angle(innovation[2])
        cross = self.covariance @ observation.T
        innovation_covariance = observation @ cross + noise
        gain = np.linalg.solve(innovation_covariance, cross.T).T
        self.state += gain @ innovation
        if self.with_angle:
            self.state[2] = wrap_angle(self.state[2])
        residual = np.eye(dimensions) - gain @ observation
        self.covariance = (
            residual @ self.covariance @ residual.T + gain @ noise @ gain.T
        )
        self.covariance = (self.covariance + self.covariance.T) * 0.5
        return self.state.copy()
