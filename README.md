# 地形匹配基线库

提供使用像素坐标的 TERCOM、ICCP、粒子滤波（PF）和点质量滤波（PMF）定位器，以及轨迹与定位误差可视化。

- **TERCOM**：通过粗细两阶段地形深度匹配估计平移。
- **ICCP**：通过等深线配准估计平移与旋转。
- **PF**：根据 INS 位移和逐点深度观测更新位置分布。
- **PMF**：在随 INS 平移的规则网格上传播概率，并用逐点深度观测更新位置分布。

TERCOM 和 ICCP 可选开启卡尔曼滤波（KF）；四种方法使用统一接口，每次调用独立初始化状态，PF 按配置种子复现结果，PMF 不使用随机采样。

## 快速开始

支持 Python 3.10+。在其他项目中使用：

1. 获取源码到目标项目的 `vendor/terrain-matching-foundation/`，确保该目录直接包含 `pyproject.toml`：

   - Windows：打开 [GitHub 仓库](https://github.com/Zhoushen-W/terrain-matching-foundation)，选择 **Code → Download ZIP**，解压后重命名为 `terrain-matching-foundation`。
   - Linux：

     ```shell
     mkdir -p vendor && cd vendor
     git clone https://github.com/Zhoushen-W/terrain-matching-foundation.git
     cd terrain-matching-foundation && rm -rf .git
     ```

     删除 `.git` 后，包内改动不会被自身的 Git 跟踪，修改包内代码不受限制。
2. 在目标项目的 `.gitignore` 中添加：

   ```gitignore
   /vendor/terrain-matching-foundation/
   ```

3. 使用目标项目的 Python 环境，在目标项目根目录执行：

   ```shell
   python -m pip install -e ./vendor/terrain-matching-foundation
   ```

4. 准备输入数组后，在自己的代码中导入并调用：

   ```python
   from terrain_matching_foundation import PMFLocator

   locator = PMFLocator(terrain_map, config_path="config.yaml")
   corrected = locator.localize_trajectory(
       reference_trajectory, ins_trajectory, depth_observations
   )
   ```

`config.yaml` 使用下方的最小配置；省略 `config_path` 则使用库默认值。源码保留在下载目录中，自己的接口文件和配置放在该目录之外，正常跟随目标项目的 Git 管理。移动下载目录后需重新安装。

将 `PMFLocator` 替换为 `TercomLocator`、`IccpLocator` 或 `PFLocator` 即可切换方法，均从 `terrain_matching_foundation` 导入。其他工具通过其 `utils` 子包访问，绘图入口见下方示例。

## 输入与输出

准备以下 NumPy 数组：

| 输入 | 形状 | 含义 |
|---|---|---|
| `terrain_map` | `(H, W)`，至少 2×2 | 地形深度，NaN 表示无效区域 |
| `reference_trajectory` | `(N, 2)` | 参考轨迹，仅用于外部评估，不参与定位 |
| `ins_trajectory` | `(N, 2)` | 原始 INS 轨迹 |
| `depth_observations` | `(N,)` | 与轨迹逐点对应的深度观测 |

轨迹每行为 `[x, y]`，地图按 `map[y, x]` 访问，整数坐标对应像素中心。位置与搜索半径使用像素，地形与观测须使用相同的深度单位、正负号和基准。除地形中的 NaN 外，输入应为有限实数；类型与基本数值性质由调用方保证。

窗口长度默认 `L=64`，仅支持 `stride=1`。输出为形状 `(max(N-L+1, 0), 2)` 的 `float64` 数组，对应原轨迹前缀；`N<L` 时返回空数组。TERCOM、ICCP 使用前向窗口，PF、PMF 仅使用当前及历史观测；对后两者设置 `window_size: 1` 可处理完整序列。

输入数组不会被修改。参考轨迹的数值不参与定位；每次调用独立初始化，不自动延续上次调用的滤波状态。

## 配置

将以下配置保存到目标项目根目录的 `config.yaml`，让 PMF 处理完整序列：

```yaml
common:
  window_size: 1
pmf:
  init_radius: 32.0
  grid_step: 0.5
  process_std: 0.5
  observation_std: 1.0
```

自定义 YAML 只需填写要覆盖的字段，其余沿用 [默认配置](terrain_matching_foundation/baselines/config.yaml)。半径、网格间距和每步过程标准差使用像素，观测标准差使用深度单位，需按实际数据调整。PMF 网格范围和间距固定，不自动扩展或细化。

TERCOM、ICCP 可在各自配置节设置 `use_kf: true`；PF 可设置 `seed` 复现结果。无有效地图支持或匹配失败时保留先验并输出日志；输出位置不保证在地图内。

## 可视化

使用与输入像素坐标对应、带 CRS 和仿射变换的地形 TIFF，绘制轨迹和算法位置误差：

```python
from terrain_matching_foundation import plot_trajectory_comparison

trajectory_figure, error_figure = plot_trajectory_comparison(
    reference_trajectory,
    ins_trajectory,
    terrain_path="data/terrain.tif",
    resolution=30.0,
    algorithm_trajectories={"PMF": corrected},
    output_dir="outputs/figures",
)
```

`resolution` 为实际米/像素比例，示例值需自行替换；它只用于误差换算。参考与 INS 轨迹须同长度且非空，各算法轨迹不得长于参考轨迹，并与其前缀对齐。

默认时间间隔为 `dt=15.0` 秒，可通过 `timestamps` 提供实际时间。指定输出目录后保存轨迹图和误差图，设 `show=True` 可显示图像。完整参数见 [绘图入口](terrain_matching_foundation/scripts/visualize_trajectories.py)。
