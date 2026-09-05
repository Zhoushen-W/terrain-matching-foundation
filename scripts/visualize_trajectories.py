"""绘制地理参考底图上的轨迹，以及各算法相对参考轨迹的位置误差。"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D

from utils.visualization import (
    algorithm_colors,
    center_trajectory_axes,
    coordinate_labels,
    elapsed_seconds,
    pixel_to_map_coordinates,
    position_errors,
    positive_scalar,
    read_georeferenced_terrain,
    validate_plot_trajectories,
)


def _plot_trajectories(
    terrain, transform, crs, depth_label, reference, ins, algorithms, colors
):
    """在 TIFF 原始坐标系中绘制完整轨迹和地形色条。

    Args:
        terrain: 已掩蔽无效值的地形波段。
        transform: TIFF 的完整仿射变换。
        crs: TIFF 的坐标参考系。
        depth_label: 地形色条标签。
        reference: 已转换到地图坐标系的参考轨迹。
        ins: 已转换到地图坐标系的 INS 轨迹。
        algorithms: 已转换到地图坐标系的算法轨迹词典。
        colors: 两张图共用的算法颜色映射。

    Returns:
        matplotlib.figure.Figure: 独立的轨迹图。
    """
    figure, axes = plt.subplots(figsize=(8, 7), layout="constrained")
    height, width = terrain.shape
    image_transform = Affine2D.from_values(
        transform.a, transform.d, transform.b, transform.e, transform.c, transform.f
    )
    # 底图坐标描述像素边界，轨迹坐标描述像素中心。
    image = axes.imshow(
        terrain,
        cmap="terrain",
        origin="upper",
        extent=(0, width, height, 0),
        interpolation="nearest",
        transform=image_transform + axes.transData,
        zorder=0,
    )
    colorbar = figure.colorbar(image, ax=axes, fraction=0.05, pad=0.04)
    colorbar.set_label(depth_label)
    axes.plot(
        reference[:, 0],
        reference[:, 1],
        color="black",
        linestyle="-",
        linewidth=3,
        label="Reference",
        zorder=3,
        marker="o" if len(reference) == 1 else None,
        markersize=5,
    )
    axes.plot(
        ins[:, 0],
        ins[:, 1],
        color="0.5",
        linestyle="--",
        linewidth=2.5,
        label="INS",
        zorder=3,
        marker="o" if len(ins) == 1 else None,
        markersize=4,
    )
    for name, trajectory in algorithms.items():
        if len(trajectory):
            axes.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                color=colors[name],
                linestyle="-",
                linewidth=1.2,
                label=name,
                zorder=4,
                marker="o" if len(trajectory) == 1 else None,
                markersize=3,
            )
    center_trajectory_axes(axes, [reference, ins, *algorithms.values()], transform)
    xlabel, ylabel = coordinate_labels(crs)
    axes.set(xlabel=xlabel, ylabel=ylabel)
    axes.grid(False, which="both")
    axes.legend(loc="best")
    return figure


def _plot_errors(errors, times, colors):
    """绘制各算法的位置误差曲线，保持与轨迹图一致的配色。

    Args:
        errors: 非空算法轨迹对应的米制误差词典。
        times: 参考轨迹各点相对首点的经过秒数。
        colors: 两张图共用的算法颜色映射。

    Returns:
        matplotlib.figure.Figure: 独立的误差图。
    """
    figure, axes = plt.subplots(figsize=(8, 4.5), layout="constrained")
    for name, error in errors.items():
        axes.plot(
            times[: len(error)],
            error,
            color=colors[name],
            linestyle="-",
            linewidth=1.2,
            label=name,
            marker="o" if len(error) == 1 else None,
            markersize=3,
        )
    if errors:
        axes.legend(loc="best")
        axes.set_ylim(bottom=0)
        end = max(times[len(error) - 1] for error in errors.values())
        axes.set_xlim(0, end if end > 0 else 1)
    else:
        axes.text(
            0.5,
            0.5,
            "No algorithm trajectories provided",
            transform=axes.transAxes,
            ha="center",
            va="center",
        )
    axes.set(xlabel="Time (s)", ylabel="Position error (m)")
    axes.grid(False, which="both")
    return figure


def plot_trajectory_comparison(
    reference_trajectory,
    ins_trajectory,
    terrain_path,
    resolution,
    algorithm_trajectories=None,
    *,
    timestamps=None,
    dt=15.0,
    band=1,
    output_dir=None,
    show=False,
):
    """生成轨迹和算法位置误差两张独立图，可选择保存或显示。

    Args:
        reference_trajectory: 非空参考像素轨迹，形状为 (N, 2)，列为 [x, y]。
        ins_trajectory: 同长度的 INS 像素轨迹，形状为 (N, 2)。
        terrain_path: 具有有效 CRS 和仿射变换的地形 TIFF 路径。
        resolution: 正的米/像素比例，仅用于像素误差换算为米。
        algorithm_trajectories: 可选的名称到 NumPy 轨迹的词典；每条轨迹
            形状为 (M, 2)，与参考前 M 点对齐，要求 M 不超过 N。
        timestamps: 可选的长度为 N 的数值秒或 datetime64 时间戳数组。
        dt: 未传时间戳时的采样间隔，默认 15 秒。
        band: TIFF 波段编号，默认读取第 1 波段。
        output_dir: 可选输出目录，保存两张 300 DPI PNG；同名文件覆盖。
        show: 是否通过 Matplotlib 显示两张图，默认不显示。

    Returns:
        tuple: (trajectory_figure, error_figure)，供调用方继续编辑或关闭。

    Raises:
        ValueError: 轨迹形状、时间戳形状或顺序、地理参考或波段不合法。
        rasterio.errors.RasterioIOError: TIFF 无法读取。
        OSError: 输出目录或图片无法写入。
    """
    reference, ins, algorithms = validate_plot_trajectories(
        reference_trajectory, ins_trajectory, algorithm_trajectories
    )
    resolution = positive_scalar(resolution, "resolution")
    times = elapsed_seconds(len(reference), timestamps, dt)
    colors = algorithm_colors(algorithms)
    errors = position_errors(reference, algorithms, resolution)
    terrain, transform, crs, depth_label = read_georeferenced_terrain(
        terrain_path, band
    )
    reference_map = pixel_to_map_coordinates(reference, transform)
    ins_map = pixel_to_map_coordinates(ins, transform)
    algorithm_maps = {
        name: pixel_to_map_coordinates(trajectory, transform)
        for name, trajectory in algorithms.items()
    }
    trajectory_figure = _plot_trajectories(
        terrain,
        transform,
        crs,
        depth_label,
        reference_map,
        ins_map,
        algorithm_maps,
        colors,
    )
    error_figure = _plot_errors(errors, times, colors)
    if output_dir is not None:
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        trajectory_figure.savefig(
            directory / "trajectories.png", dpi=300, bbox_inches="tight"
        )
        error_figure.savefig(
            directory / "trajectory_errors.png", dpi=300, bbox_inches="tight"
        )
    if show:
        plt.show()
    return trajectory_figure, error_figure
