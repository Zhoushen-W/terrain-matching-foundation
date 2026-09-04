"""提供轨迹绘图共用的输入校验、地理坐标、时间和配色处理。"""

import warnings
from collections.abc import Mapping
from numbers import Integral, Real

import matplotlib as mpl
import numpy as np
import rasterio
from matplotlib.ticker import ScalarFormatter
from rasterio.errors import NotGeoreferencedWarning

from utils.inputs import real_array


def positive_scalar(value, name):
    """校验绘图所需的正数参数。

    Args:
        value: 待校验的数值。
        name: 用于错误信息的参数名称。

    Returns:
        float: 有限的正数。

    Raises:
        ValueError: 参数不是有限正数。
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite positive number")
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return value


def _validate_path(trajectory, name):
    """校验一条用于绘图的像素轨迹。

    Args:
        trajectory: 待校验的 NumPy 轨迹。
        name: 用于错误信息的轨迹名称。

    Returns:
        numpy.ndarray: 形状为 (M, 2) 的有限 float64 轨迹。

    Raises:
        ValueError: 轨迹类型、形状或数值不合法。
    """
    trajectory = real_array(trajectory, name, 2)
    if trajectory.shape[1] != 2 or not np.isfinite(trajectory).all():
        raise ValueError(f"{name} must have shape (M, 2) and contain finite values")
    return trajectory


def validate_plot_trajectories(reference, ins, algorithms=None):
    """校验同长度参考与 INS，以及从起点对齐的算法轨迹词典。

    Args:
        reference: 非空参考轨迹，形状为 (N, 2)。
        ins: 与参考轨迹同长度的 INS 轨迹。
        algorithms: 可选的算法名称到 NumPy 轨迹的映射。

    Returns:
        tuple: float64 参考轨迹、INS 轨迹及保持输入顺序的算法词典。

    Raises:
        ValueError: 输入形状、名称或长度不合法。
    """
    reference = _validate_path(reference, "reference_trajectory")
    ins = _validate_path(ins, "ins_trajectory")
    if len(reference) == 0 or ins.shape != reference.shape:
        raise ValueError(
            "Reference and INS trajectories must share a nonempty (N, 2) shape"
        )
    if algorithms is None:
        algorithms = {}
    if not isinstance(algorithms, Mapping):
        raise ValueError("algorithm_trajectories must map names to NumPy trajectories")
    validated = {}
    for name, trajectory in algorithms.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Algorithm names must be nonempty strings")
        trajectory = _validate_path(trajectory, name)
        if len(trajectory) > len(reference):
            raise ValueError(
                f"Algorithm {name!r} is longer than the reference trajectory"
            )
        validated[name] = trajectory
    return reference, ins, validated


def elapsed_seconds(length, timestamps=None, dt=15.0):
    """生成经过秒数，并统一校验转换结果的有限性与严格递增性。

    Args:
        length: 已校验的非空参考轨迹的长度。
        timestamps: 可选的长度为 length 的数值秒或 datetime64 数组。
        dt: 未提供时间戳时的采样间隔，单位为秒。

    Returns:
        numpy.ndarray: 长度为 length 的 float64 时间数组，从零开始。

    Raises:
        ValueError: 时间戳形状、类型、有限性或递增顺序不合法。
    """
    if timestamps is None:
        dt = positive_scalar(dt, "dt")
        with np.errstate(over="ignore"):
            elapsed = np.arange(length, dtype=np.float64) * dt
    else:
        values = np.asarray(timestamps)
        if values.shape != (length,):
            raise ValueError("timestamps must have shape (N,)")
        if values.dtype.kind == "M":
            if np.datetime_data(values.dtype)[0] in {"Y", "M"}:
                values = values.astype("datetime64[D]")
            elapsed = (values - values[0]) / np.timedelta64(1, "s")
        elif values.dtype.kind in "iuf":
            values = values.astype(np.float64)
            with np.errstate(over="ignore", invalid="ignore"):
                elapsed = values - values[0]
        else:
            raise ValueError(
                "timestamps must contain numeric seconds or datetime64 values"
            )
    elapsed = np.asarray(elapsed, dtype=np.float64)
    if not np.isfinite(elapsed).all() or np.any(np.diff(elapsed) <= 0):
        raise ValueError("Elapsed timestamps must be finite and strictly increasing")
    return elapsed


def algorithm_colors(names):
    """按算法输入顺序分配两张图共用的分类颜色。

    Args:
        names: 按显示顺序排列的算法名称。

    Returns:
        dict: 算法名称到 RGBA 颜色的映射，避开参考黑色和 INS 灰色。
    """
    names = list(names)
    palette = [
        color for index, color in enumerate(mpl.colormaps["tab10"].colors) if index != 7
    ]
    if len(names) > len(palette):
        palette = mpl.colormaps["hsv"](np.linspace(0, 1, len(names), endpoint=False))
    return {
        name: mpl.colors.to_rgba(palette[index]) for index, name in enumerate(names)
    }


def read_georeferenced_terrain(terrain_path, band=1):
    """读取具有 CRS 与有效仿射变换的 TIFF 地形波段。

    Args:
        terrain_path: TIFF 文件路径。
        band: 从 1 开始的波段编号。

    Returns:
        tuple: 掩蔽无效值后的地形、仿射变换、CRS 和深度色条标签。

    Raises:
        ValueError: 文件类型、地理信息、波段或有效地形数据不合法。
        rasterio.errors.RasterioIOError: 无法读取输入文件。
    """
    if isinstance(band, (bool, np.bool_)) or not isinstance(band, Integral):
        raise ValueError("band must be an integer starting at 1")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", NotGeoreferencedWarning)
            with rasterio.open(terrain_path) as dataset:
                if dataset.driver != "GTiff":
                    raise ValueError("terrain_path must reference a TIFF raster")
                if band < 1 or band > dataset.count:
                    raise ValueError(f"band must be between 1 and {dataset.count}")
                transform, crs = dataset.transform, dataset.crs
                if crs is None:
                    raise ValueError(
                        "The TIFF must define a coordinate reference system"
                    )
                if not np.isfinite(tuple(transform)).all() or transform.is_degenerate:
                    raise ValueError(
                        "The TIFF must define a finite invertible affine transform"
                    )
                if transform.is_identity and (dataset.gcps[0] or dataset.rpcs):
                    raise ValueError(
                        "GCP/RPC-only TIFFs require an affine georeference"
                    )
                terrain = dataset.read(int(band), masked=True)
                if terrain.dtype.kind not in "iuf":
                    raise ValueError(
                        "The terrain band must contain real numeric depths"
                    )
                terrain = terrain.astype(np.float64)
                scale, offset = dataset.scales[band - 1], dataset.offsets[band - 1]
                if not np.isfinite([scale, offset]).all():
                    raise ValueError("The terrain band scale and offset must be finite")
                with np.errstate(over="ignore", invalid="ignore"):
                    terrain = np.ma.masked_invalid(terrain * scale + offset)
                unit = dataset.units[band - 1]
    except NotGeoreferencedWarning as error:
        raise ValueError(
            "The TIFF must define a georeferencing affine transform"
        ) from error
    if terrain.count() == 0:
        raise ValueError("The terrain band contains no finite valid data")
    label = f"Depth ({unit})" if unit else "Depth"
    return terrain, transform, crs, label


def pixel_to_map_coordinates(trajectory, transform):
    """将像素中心轨迹转换到 TIFF 的原始坐标系。

    Args:
        trajectory: 已校验的形状为 (M, 2) 的像素轨迹，列顺序为 [x, y]。
        transform: 已校验的 TIFF 仿射变换，基于像素边界坐标。

    Returns:
        numpy.ndarray: 形状为 (M, 2) 的地图坐标，不修改输入。
    """
    centers = trajectory + 0.5
    return np.column_stack(
        (
            transform.a * centers[:, 0] + transform.b * centers[:, 1] + transform.c,
            transform.d * centers[:, 0] + transform.e * centers[:, 1] + transform.f,
        )
    )


def coordinate_labels(crs):
    """根据 TIFF 原始坐标系选择坐标轴名称与单位。

    Args:
        crs: TIFF 的 Rasterio 坐标参考系。

    Returns:
        tuple: 横轴和纵轴标签。
    """
    if crs.is_geographic:
        return "Longitude (deg)", "Latitude (deg)"
    if crs.is_projected:
        unit = crs.linear_units
        if unit in {"metre", "meter", "metres", "meters"}:
            unit = "m"
        return f"X ({unit})", f"Y ({unit})"
    return "X", "Y"


def center_trajectory_axes(axes, trajectories, transform):
    """以完整轨迹包围盒为中心设置等比例地图显示范围。

    Args:
        axes: 轨迹图坐标轴。
        trajectories: 地图坐标轨迹序列，至少一条非空。
        transform: TIFF 仿射变换，用于确定退化轨迹的最小显示跨度。

    Returns:
        None: 设置坐标范围、比例与科学计数格式。
    """
    nonempty = [trajectory for trajectory in trajectories if len(trajectory)]
    lower = np.min([trajectory.min(axis=0) for trajectory in nonempty], axis=0)
    upper = np.max([trajectory.max(axis=0) for trajectory in nonempty], axis=0)
    center = lower + (upper - lower) * 0.5
    span = max(
        float(np.max(upper - lower)),
        float(np.hypot(transform.a, transform.d)),
        float(np.hypot(transform.b, transform.e)),
    )
    half_span = span * 0.6
    axes.set_xlim(center[0] - half_span, center[0] + half_span)
    axes.set_ylim(center[1] - half_span, center[1] + half_span)
    axes.set_aspect("equal", adjustable="box", anchor="C")
    for axis in (axes.xaxis, axes.yaxis):
        formatter = ScalarFormatter(useOffset=False, useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((0, 0))
        axis.set_major_formatter(formatter)


def position_errors(reference, algorithms, resolution):
    """分别计算算法轨迹相对参考前缀的米制位置误差。

    Args:
        reference: 已校验的形状为 (N, 2) 的参考轨迹。
        algorithms: 已校验的算法轨迹词典。
        resolution: 已校验的有限正数，单位为米/像素。

    Returns:
        dict: 非空算法轨迹对应的一维米制误差数组。
    """
    errors = {}
    for name, trajectory in algorithms.items():
        if len(trajectory) == 0:
            continue
        difference = trajectory - reference[: len(trajectory)]
        error = np.hypot(difference[:, 0], difference[:, 1]) * resolution
        errors[name] = error
    return errors
