"""运行固定合成场景并报告三种基线及 KF 开关的定位精度。"""

import argparse
import json
import tempfile
import time
import tracemalloc
from pathlib import Path

import numpy as np
import yaml

from baselines import IccpLocator, PFLocator, TercomLocator
from tests.support import synthetic_data
from utils.geometry import rotation_matrix


def error_metrics(corrected, reference):
    """计算相同轨迹前缀上的像素位置误差统计。

    Args:
        corrected: 非空修正轨迹。
        reference: 完整参考轨迹。

    Returns:
        dict: RMSE、均值、95%分位数及最大误差，单位为像素。
    """
    errors = np.linalg.norm(corrected - reference[: len(corrected)], axis=1)
    return {
        "rmse_px": float(np.sqrt(np.mean(errors**2))),
        "mean_px": float(np.mean(errors)),
        "p95_px": float(np.quantile(errors, 0.95)),
        "max_px": float(np.max(errors)),
    }


def accuracy_cases():
    """生成与数值单测使用不同随机种子的固定验收场景。

    Args:
        无。

    Returns:
        list: 含地图、真值、INS、观测及生成参数的场景字典。
    """
    field, reference, clean_depths = synthetic_data(count=80, seed=20260904)
    cases = []
    specifications = (
        ("integer_shift", [3.0, -2.0], 0.0, 0.0, False),
        ("subpixel_shift", [2.75, -1.25], 0.0, 0.0, False),
        ("heading_error", [2.75, -1.25], 2.0, 0.0, False),
        ("drift_low_noise", [2.75, -1.25], 0.0, 0.25, True),
        ("drift_high_noise", [2.75, -1.25], 0.0, 1.0, True),
    )
    noise = np.random.default_rng(20260905).normal(size=len(reference))
    for name, bias, angle, sigma, drift in specifications:
        ins = (
            (reference - reference[0]) @ rotation_matrix(-np.deg2rad(angle)).T
            + reference[0]
            + bias
        )
        if drift:
            ins += np.linspace(0, 1, len(ins))[:, None] * [2.0, -1.0]
        cases.append(
            {
                "name": name,
                "field": field,
                "reference": reference,
                "ins": ins,
                "depths": clean_depths + sigma * noise,
                "noise_std": sigma,
                "angle_deg": None if drift else angle,
            }
        )
    return cases


def evaluation_config(case):
    """建立合成场景的统一先验与观测噪声配置。

    Args:
        case: 合成场景描述。

    Returns:
        dict: 覆盖默认配置的评估参数；算法不能读取参考轨迹数值。
    """
    return {
        "common": {"window_size": 32},
        "tercom": {"coarse_radius": 8, "coarse_step": 1},
        "iccp": {"init_radius": 8, "init_step": 1},
        "pf": {
            "init_radius": 8,
            "process_std": 0.2,
            "observation_std": max(0.2, case["noise_std"]),
        },
    }


def run_evaluation(trace_memory=False):
    """运行五类固定场景及多个 PF 种子的对照实验。

    Args:
        trace_memory: 是否记录 tracemalloc 峰值；启用会影响运行时间。

    Returns:
        dict: 可序列化的配置和逐次实验结果。
    """
    root = Path(__file__).resolve().parents[1] / ".codex_tmp"
    root.mkdir(exist_ok=True)
    records = []
    configurations = {}
    with tempfile.TemporaryDirectory(dir=root) as directory:
        path = Path(directory) / "evaluation.yaml"
        for case in accuracy_cases():
            base = evaluation_config(case)
            configurations[case["name"]] = base
            count = len(case["ins"]) - base["common"]["window_size"] + 1
            records.append(
                {
                    "case": case["name"],
                    "method": "INS",
                    "seed": None,
                    **error_metrics(case["ins"][:count], case["reference"]),
                }
            )
            variants = [
                (TercomLocator, False, None),
                (TercomLocator, True, None),
                (IccpLocator, False, None),
                (IccpLocator, True, None),
            ] + [(PFLocator, False, seed) for seed in range(5)]
            for locator_type, use_kf, seed in variants:
                config = evaluation_config(case)
                config["tercom"]["use_kf"] = use_kf
                config["iccp"]["use_kf"] = use_kf
                if seed is not None:
                    config["pf"]["seed"] = seed
                path.write_text(yaml.safe_dump(config), encoding="utf-8")
                if trace_memory:
                    tracemalloc.start()
                begin = time.perf_counter()
                locator = locator_type(case["field"], path)
                corrected = locator.localize_trajectory(
                    case["reference"], case["ins"], case["depths"]
                )
                elapsed = time.perf_counter() - begin
                peak = None
                if trace_memory:
                    _, peak = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                record = {
                    "case": case["name"],
                    "method": locator_type.__name__ + ("+KF" if use_kf else ""),
                    "seed": seed,
                    "elapsed_seconds": elapsed,
                    "traced_peak_mib": None if peak is None else peak / (1024**2),
                    **error_metrics(corrected, case["reference"]),
                }
                if locator_type is IccpLocator and case["angle_deg"] is not None:
                    record["final_angle_error_deg"] = float(
                        abs(np.rad2deg(locator.transform_[2]) - case["angle_deg"])
                    )
                records.append(record)
    return {
        "map_seed": 20260904,
        "observation_seed": 20260905,
        "config_overrides": configurations,
        "records": records,
    }


def main():
    """解析命令行并输出可复现的精度实验报告。

    Args:
        无。

    Returns:
        None: 输出 JSON 到标准输出或指定文件。
    """
    parser = argparse.ArgumentParser(
        description="Evaluate synthetic localization accuracy"
    )
    parser.add_argument("--output", type=Path, help="JSON output path")
    parser.add_argument(
        "--trace-memory", action="store_true", help="Trace Python allocation peaks"
    )
    args = parser.parse_args()
    report = json.dumps(run_evaluation(args.trace_memory), indent=2, allow_nan=False)
    if args.output is None:
        print(report)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n", encoding="utf-8")
        print(f"Accuracy report written to {args.output}")


if __name__ == "__main__":
    main()
