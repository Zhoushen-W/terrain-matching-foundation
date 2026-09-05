# 地形匹配基线库

提供使用像素坐标的 TERCOM、ICCP 和粒子滤波（PF）定位器，以及轨迹与定位误差可视化。

- **TERCOM**：通过粗细两阶段地形深度匹配估计平移。
- **ICCP**：通过等深线配准估计平移与旋转。
- **PF**：根据 INS 位移和逐点深度观测更新位置分布。

TERCOM 和 ICCP 可选开启卡尔曼滤波（KF）；三种方法使用统一接口，每次调用独立初始化状态，PF 按配置种子复现结果。

## 安装

支持 Python 3.10+，项目开发使用 Conda `Terrain` 环境（Python 3.12.14）。在项目根目录安装运行依赖和 Ruff：

```powershell
& 'E:\Anaconda\envs\Terrain\python.exe' -m pip install -e '.[dev]'
```

## 快速开始

准备以下 NumPy 数组：

| 输入 | 形状 | 含义 |
|---|---|---|
| `terrain_map` | `(H, W)`，至少 2×2 | 地形深度，NaN 表示无效区域 |
| `reference_trajectory` | `(N, 2)` | 参考轨迹，仅用于外部评估，不参与定位 |
| `ins_trajectory` | `(N, 2)` | 原始 INS 轨迹 |
| `depth_observations` | `(N,)` | 与轨迹逐点对应的深度观测 |

轨迹每行为 `[x, y]`，地图按 `map[y, x]` 访问，整数坐标对应像素中心。位置与搜索半径使用像素，地形与观测须使用相同的深度单位、正负号和基准。除地形中的 NaN 外，输入应为有限实数；类型与基本数值性质由调用方保证。

```python
from baselines import TercomLocator

locator = TercomLocator(terrain_map, config_path=None)
corrected = locator.localize_trajectory(
    reference_trajectory, ins_trajectory, depth_observations
)
reference_aligned = reference_trajectory[: len(corrected)]
```

将 `TercomLocator` 替换为 `IccpLocator` 或 `PFLocator` 即可切换方法，均从 `baselines` 导入。输入数组不会被修改。

窗口长度默认 `L=64`，仅支持 `stride=1`。输出为形状 `(max(N-L+1, 0), 2)` 的 `float64` 数组，对应原轨迹前缀；`N<L` 时返回空数组。TERCOM、ICCP 使用前向窗口，PF 仅使用当前及历史观测。

## 配置

完整参数及默认值见 [baselines/config.yaml](baselines/config.yaml)。自定义 YAML 可只包含需要覆盖的字段，例如保存为 `config.yaml`：

```yaml
common:
  window_size: 32
tercom:
  use_kf: true
pf:
  observation_std: 0.5
  seed: 42
```

通过 `TercomLocator(terrain_map, config_path="config.yaml")` 加载；未填写的字段沿用默认值。调参时重点关注窗口长度、搜索半径与间距、深度观测噪声和粒子数量。

默认参数用于启动实验，实际精度需结合地形与噪声验证。无有效地图支持或匹配失败时，定位器保留先验并输出日志；输出位置不保证在地图内。

## 可视化

使用与输入像素坐标对应、带 CRS 和仿射变换的地形 TIFF，绘制轨迹和算法位置误差：

```python
from scripts.visualize_trajectories import plot_trajectory_comparison

trajectory_figure, error_figure = plot_trajectory_comparison(
    reference_trajectory,
    ins_trajectory,
    terrain_path="data/terrain.tif",
    resolution=30.0,
    algorithm_trajectories={"TERCOM": corrected},
    output_dir="outputs/figures",
)
```

`resolution` 为实际米/像素比例，示例值需自行替换；它只用于误差换算。参考与 INS 轨迹须同长度且非空，各算法轨迹不得长于参考轨迹，并与其前缀对齐。

默认时间间隔 `dt=15.0` 秒，也可传入长度为 N、严格递增的数值秒或 `datetime64` 时间戳 `timestamps`。指定输出目录后保存 `trajectories.png` 和 `trajectory_errors.png`，同名文件会覆盖；设 `show=True` 可显示图像。完整参数见 [绘图入口](scripts/visualize_trajectories.py)。

## 验证与目录

```powershell
& 'E:\Anaconda\envs\Terrain\python.exe' -X utf8 -m unittest discover -s tests -t .
& 'E:\Anaconda\envs\Terrain\python.exe' -m ruff check .
& 'E:\Anaconda\envs\Terrain\python.exe' -m ruff format --check .
```

生成合成数据精度报告：

```powershell
& 'E:\Anaconda\envs\Terrain\python.exe' -X utf8 -m tests.evaluate_accuracy --output .codex_tmp/accuracy.json
```

报告覆盖平移、旋转、漂移及噪声场景，用于精度回归；真实数据上的参数与效果需独立评估。

```text
baselines/   定位器与默认配置
utils/       配置、数值计算和绘图工具
scripts/     可导入的绘图入口
tests/       单元测试与合成精度实验
.codex_tmp/  临时配置与实验输出
```