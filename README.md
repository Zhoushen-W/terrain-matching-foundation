# 地形匹配基线库

本仓库提供独立的 TERCOM、ICCP 和标准粒子滤波（PF）定位器，以及轨迹与定位误差可视化。定位输入和输出使用像素坐标。实现以算法正确性、定位精度及结果可复现性为首要目标。

## 安装与调用

代码支持 Python 3.10 或以上，项目开发使用 Conda 的 `Terrain` 环境（Python 3.12）。在项目根目录执行以下 PowerShell 命令安装运行依赖及 Ruff：

```powershell
& 'E:\Anaconda\envs\Terrain\python.exe' -m pip install -r requirements.txt
```

`requirements.txt` 与 `pyproject.toml` 中的运行依赖及 `dev` 依赖保持一致。运行依赖为 NumPy、SciPy、PyYAML、ContourPy、Matplotlib 和 Rasterio；开发依赖 Ruff，基线测试使用标准库 `unittest`。Python 命令始终使用该环境的解释器路径，避免子 shell 中激活环境不生效。

若需在其他目录导入本库，可在该环境中进行可编辑安装：

```powershell
& 'E:\Anaconda\envs\Terrain\python.exe' -m pip install -e .
```

```python
from baselines import IccpLocator, PFLocator, TercomLocator

locator = TercomLocator(terrain_map, config_path=None)
corrected = locator.localize_trajectory(
    reference_trajectory,
    ins_trajectory,
    depth_observations,
)
```

替换定位器类即可运行其他方法。三个类继承相同的构造接口。每次调用独立初始化递推状态；PF 每次调用按配置种子重新初始化随机数生成器。

## 数据约定

| 输入 | 类型和形状 | 含义 |
|---|---|---|
| `terrain_map` | 实数 NumPy 数组 `(H, W)` | 地形深度，允许 NaN 表示无效区域 |
| `reference_trajectory` | 实数 NumPy 数组 `(N, 2)` | 外部评估参考，不参与定位计算 |
| `ins_trajectory` | 实数 NumPy 数组 `(N, 2)` | 原始 INS 轨迹 |
| `depth_observations` | 实数 NumPy 数组 `(N,)` | 实际输入的深度观测 |

轨迹列顺序是 `[x, y]`，地图按 `map[y, x]` 访问。整数坐标对应像素中心，浮点坐标使用双线性插值。有效采样范围为 `0 ≤ x ≤ W-1`、`0 ≤ y ≤ H-1`。

地图与观测须具有相同的深度单位、正负号和基准。所有位置、搜索半径及位置噪声均使用像素；内部旋转使用弧度，KF 配置中的角度标准差使用度。像素 y 轴向下时，正旋转角在图上表现为顺时针。

轨迹与观测必须有限且形状一致，参考轨迹也执行这些输入校验，但其位置数值不参与初始化、匹配或观测生成。地形至少为 2×2，不允许正负无穷。输入不会被修改，定位器持有独立只读地形副本。

## 窗口与输出

首版仅支持 `stride=1`。窗口长度为 `L`，起点依次为 `s=0,…,N-L`。

- TERCOM 和 ICCP 使用完整窗口 `[s:s+L]`，只提交该窗口起点的修正位置。
- PF 每次只处理当前点的 INS 增量及一条深度观测，运行至 `t=N-L`。它不读取未来观测，也不重复使用重叠窗口观测。
- 返回形状为 `(max(N-L+1, 0), 2)` 的独立 `float64` 数组。`N<L` 时返回空数组。

例如 `N=10、L=4`，返回原索引 0～6 的七个修正点，原索引 7～9 舍弃。比较误差时使用 `reference_trajectory[:len(corrected)]`。

三种方法的输出区间相同，但 TERCOM、ICCP 对窗口起点的估计使用后续 `L-1` 条观测，PF 是因果逐点估计。实验中应保留这一观测信息量差异的说明。

## 配置

默认配置位于 `baselines/config.yaml`。`config_path` 可传入完整 YAML 或局部覆盖文件，未指定字段保留默认值。未知字段、非有限数值和非法范围会明确报错。基础类型遵循默认配置的约定，由调用方保证，不额外检查字典、整数或布尔类型，也不对类型误用统一包装异常。

下面是一份局部覆盖示例：

```yaml
common:
  window_size: 64
tercom:
  use_kf: true
  coarse_radius: 16.0
iccp:
  use_kf: true
pf:
  observation_std: 0.5
  seed: 42
```

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `common.window_size` | 64 | 完整窗口长度 |
| `common.stride` | 1 | 首版只接受 1 |
| `common.candidate_batch_size` | 1024 | TERCOM 每批候选数，不改变搜索结果 |
| `tercom.coarse_radius / coarse_step` | 32 / 2 | 每窗粗搜半径及间距 |
| `tercom.fine_radius / fine_step` | 2 / 0.25 | 每窗精搜半径及间距 |
| `iccp.init_radius / init_step` | 32 / 2 | 首窗 TERCOM 粗初始化 |
| `iccp.max_iterations` | 50 | 每窗最多迭代次数 |
| `iccp.translation_tolerance` | 0.001 | 固定原点处平移更新的收敛阈值，像素 |
| `iccp.angle_tolerance` | 0.0001 | 角度更新的收敛阈值，弧度 |
| `tercom.use_kf / iccp.use_kf` | false | 两种方法分别控制 KF |
| `*.kalman.initial_std / process_std / measurement_std` | 32 / 0.5 / 1 | 平移初始、过程、观测标准差，像素 |
| `iccp.kalman.*_angle_std_deg` | 10 / 0.1 / 1 | 角度初始、过程、观测标准差，度 |
| `pf.num_particles` | 4096 | 粒子数量 |
| `pf.init_radius` | 32 | INS 首点周围正方形均匀先验的半宽 |
| `pf.process_std` | 0.5 | 每步各位置轴的过程标准差，像素 |
| `pf.observation_std` | 1 | 深度观测标准差，与地图数值同单位 |
| `pf.resample_threshold` | 0.5 | 有效粒子数与总粒子数的阈值比例 |
| `pf.seed` | 0 | 非负整数随机种子 |

标准差配置内部平方为协方差；`process_std` 和搜索半径允许为零。搜索轴包含中心和半径边界，半径不能整除间距时，最外侧间距缩短。

默认值用于启动实验，尚未针对真实地图调优。应先确认位置先验覆盖范围和深度噪声尺度，再用独立调参数据选择窗口长度、搜索间距及滤波参数。精度验收数据应与调参数据分开。

## 算法与递推

### TERCOM

每个窗口都对原始 INS 段执行二维平移粗搜索，再围绕粗搜索最优解精搜索。完整窗口的地图深度与观测深度平均绝对差（MAD）作为目标。相同代价下优先选取距离搜索中心更近的候选，再按坐标顺序打破剩余同分，确保分块计算可复现。

首窗以零平移为先验，后续窗口以上一窗最终平移为搜索中心。修正始终应用于原始 INS，避免重复叠加已有修正。

### ICCP

每次轨迹调用只执行一次 TERCOM 粗搜索初始化。随后根据每条观测深度，用 ContourPy 提取全图等深线，查询到连续线段（包含内部及端点）的最近投影，交替求解对应关系及二维最小二乘刚体变换。算法不引入局部对应搜索半径，也不量化观测深度。

仅保留当前窗口的等深线几何并在迭代中复用，不保留完整轨迹的复杂缓存。含 NaN 的网格单元不生成穿越缺测区域的等深线。等深线由连续线段近似，因此它与双线性地形的精确等值曲线可能存在微小离散误差。

固定原点为本次调用的 `o=ins_trajectory[0]`，完整变换为：

```text
corrected_point = o + R(theta) @ (ins_point - o) + [dx, dy]
```

刚体求解禁止反射。无法确定旋转时保持已有角度，仅使用有效对应点更新平移。平坦地形中已位于观测等深区域的点保持自身位置。窗口长度为 1 时不会凭单点推断旋转。

### 可选 KF

TERCOM 递推 `[dx, dy]`，ICCP 递推 `[dx, dy, theta]`。两者采用随机游走模型，匹配结果作为变换观测；开启时，后验变换既用于输出，也作为下一窗口先验。关闭时直接沿用匹配变换。

角度创新采用最短周期差，协方差使用 Joseph 更新。无法确定旋转时只融合平移观测。重叠窗口的观测相关性没有被完整建模，因此该 KF 是递推平滑器，其协方差不应解释为严格校准的不确定度，开启后也不保证所有场景精度更好。

TERCOM 和 ICCP 调用结束后可读取 `transform_` 获取最终变换副本；它在每次调用时重置，不改变轨迹返回格式。

### PF

粒子状态为绝对像素位置，初始中心只取 INS 首点。每步使用原始 INS 位移增量预测，使用单条深度观测的高斯似然更新对数权重。估计采用重采样前的加权均值，有效粒子数低于阈值时执行系统重采样。

该基线使用标准逐点 PF，不使用 TERCOM 初始化、窗口似然、额外粒子抖动或真值起点。

## 无效数据与失败处理

双线性采样中，正权重邻点为 NaN 才使结果无效；整数采样点不会因为零权重邻点缺失而被拒绝。越界坐标不会裁剪到地图边缘。

- TERCOM 无完整有效候选时保留先验；ICCP 无有效等深线解时保留先验。首窗已得到有效粗初始化时，ICCP 可保留该初始化。
- 开启 KF 时，无新匹配结果只做预测；ICCP 仅有粗初始化时只融合其平移。
- PF 的无效粒子在正常观测更新中获得零权重；若所有正权重粒子都没有有效地图支持，则保留预测分布及此前权重，并跳过该次重采样。
- 上述失败通过英文日志报告。先验本身越界时，输出也可能越界，应结合日志判断该段定位是否有观测支持。

## 轨迹与定位误差可视化

通过可导入脚本生成两张独立的 Matplotlib Figure：

```python
from scripts.visualize_trajectories import plot_trajectory_comparison

trajectory_figure, error_figure = plot_trajectory_comparison(
    reference_trajectory,
    ins_trajectory,
    terrain_path="data/terrain.tif",
    resolution=30.0,
    algorithm_trajectories={
        "TERCOM": tercom_trajectory,
        "ICCP": iccp_trajectory,
        "PF": pf_trajectory,
    },
    timestamps=None,
    dt=15.0,
    band=1,
    output_dir="outputs/figures",
    show=False,
)
```

示例中的 `resolution=30.0` 表示每像素 30 米，仅为演示值，请替换为实际米/像素比例。它只参与误差换算，不改变轨迹图的地理坐标。此比例为标量，适用于误差评估所采用的统一、等尺度像素坐标；它不从 TIFF 的坐标单位自动推断。

| 参数 | 约定 |
|---|---|
| `reference_trajectory`、`ins_trajectory` | 同长度、非空的有限实数 NumPy 数组 `(N, 2)`，列为 `[x, y]` 像素坐标 |
| `terrain_path` | 带有效 CRS 和仿射变换的 TIFF 文件；缺少地理信息时明确报错 |
| `resolution` | 必填有限正数，单位为米/像素 |
| `algorithm_trajectories` | 可选的 `{名称: NumPy轨迹}` 词典，各轨迹长度可以不同，但均不得超过 N |
| `timestamps` | 可选长度 N 的数值秒数组或 NumPy `datetime64` 数组，必须有效且严格递增 |
| `dt` | 无时间戳时使用的正采样间隔，默认 15 秒；有时间戳时忽略该参数 |
| `band` | 从 1 开始的 TIFF 波段编号，默认 1 |
| `output_dir` | 可选保存目录；未指定时不保存文件 |
| `show` | 默认 false；设为 true 时调用 Matplotlib 显示两张图 |

### 轨迹图

底图采用 TIFF 原始坐标系，读取指定波段并掩蔽 NoData 与非有限值；按波段的 scale、offset 恢复显示值。使用 `terrain` 色图及深度 colorbar，深度单位有明确元数据时写入标签。

输入轨迹的整数坐标代表像素中心，转换到地图坐标时使用 TIFF 仿射变换作用于 `(x+0.5, y+0.5)`。底图使用同一完整变换，因此包含旋转或剪切项的仿射地图也按照原始空间关系绘制。仅有 GCP/RPC 而没有可用仿射地理参考的文件不属于该入口的支持范围。

轨迹图坐标轴使用 TIFF 原始坐标单位，强制科学计数法并关闭额外常数偏移显示；经纬度地图标注经纬度，投影地图标注其坐标单位。显示范围根据参考、INS 及全部算法轨迹的包围盒确定，以其中心为中心保持等比例，并在最长跨度两侧各预留约 10% 空间。轨迹超出底图覆盖范围时，也会完整显示对应坐标范围。

- 参考轨迹：黑色实线，线宽 3。
- INS：灰色虚线，线宽 2.5。
- 算法轨迹：线宽 1.2 的实线，按词典顺序分配颜色，跳过灰色；算法数超过九种时均匀扩展色相配色。
- 空算法轨迹跳过；只有一个采样点的轨迹用小圆点显示，以免单点实线不可见。

### 误差图与时间

误差图只绘制算法词典中的轨迹，不绘制参考或 INS 误差。每条长度为 M 的算法轨迹与参考前 M 点逐点对齐：

```text
error[i] = sqrt((algorithm_x[i] - reference_x[i])**2
                + (algorithm_y[i] - reference_y[i])**2) * resolution
```

不进行插值、重采样或自动截断过长轨迹。算法名称和颜色与轨迹图一致，横轴为 `Time (s)`、纵轴为 `Position error (m)`，不显示 grid。没有有效算法轨迹时，误差图显示英文提示，轨迹图仍显示参考与 INS。

未提供时间戳时使用 `0, 15, 30, …` 秒，或使用指定的 `dt`。提供时间戳时统一减去首点时间，转换为从零开始的经过秒数，各算法只使用其对应的前 M 个时间点。

```python
import numpy as np

timestamps = np.array(
    [
        "2026-09-04T10:00:00",
        "2026-09-04T10:00:12",
        "2026-09-04T10:00:29",
    ],
    dtype="datetime64[s]",
)
```

此示例在三点输入中对应横坐标 `0, 12, 29` 秒。实际传入的时间戳数量须与完整参考轨迹长度一致。

指定输出目录时，保存 `trajectories.png` 和 `trajectory_errors.png`，分辨率为 300 DPI，同名文件会覆盖。返回的 Figure 可继续编辑或自行保存成 PDF、SVG 等格式；用完后由调用方通过 `matplotlib.pyplot.close` 关闭。

按照本次要求，可视化功能不新增或运行测试，不安排绘图验收及安装包验证，仅进行 Ruff 静态规范检查。

## 基线验证

```powershell
& 'E:\Anaconda\envs\Terrain\python.exe' -X utf8 -m unittest discover -s tests -t .
& 'E:\Anaconda\envs\Terrain\python.exe' -m ruff check .
& 'E:\Anaconda\envs\Terrain\python.exe' -m ruff format --check .
& 'E:\Anaconda\envs\Terrain\python.exe' -X utf8 -m tests.evaluate_accuracy --output .codex_tmp/accuracy.json
```

精度实验包括整数平移、亚像素平移、航向误差、持续漂移及两档深度噪声；对照 KF 开关并运行五个 PF 种子。报告同时保存配置覆盖、RMSE、平均误差、95%分位误差、最大误差和运行时间。

需要记录内存时增加 `--trace-memory`，报告中的 `traced_peak_mib` 是 tracemalloc 可追踪的分配峰值，不等同于操作系统进程驻留内存；开启追踪会影响计时。

实际验证结果和精度限制见 `VALIDATION.md`。当前阶段的合成数据用于工程正确性与精度回归，真实数据上的最终参数和泛化精度仍需独立实验。

## 目录与开发约定

```text
baselines/    定位器、统一接口与默认 YAML
utils/        公共配置、采样、几何、匹配及滤波工具
tests/        独立数值测试、接口测试和合成精度实验
scripts/      可导入的轨迹与定位误差绘图入口
.codex_tmp/   临时配置与实验输出，任务结束后清理其临时内容
```

遵循根目录 `AGENTS.md`：Python 文件说明、函数 Docstring 和普通代码注释使用中文；日志、异常报错信息和业务字符串使用英文。保留必要的数组、取值范围及算法有效性校验，信任调用方提供的基础数据类型。新增代码执行 Ruff 规范检查与格式检查。
