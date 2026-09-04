"""读取和校验基线的 YAML 配置，支持通过局部配置覆盖默认值。"""

from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml


def _merge_config(defaults, overrides, prefix=""):
    """递归合并配置并拒绝未知字段。

    Args:
        defaults: 默认配置字典。
        overrides: 覆盖配置字典。
        prefix: 当前字段路径。

    Returns:
        dict: 合并后的独立配置。

    Raises:
        ValueError: 配置结构或字段不合法。
    """
    if not isinstance(overrides, dict):
        raise ValueError(f"Configuration section {prefix or 'root'} must be a mapping")
    result = deepcopy(defaults)
    for key, value in overrides.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if key not in defaults:
            raise ValueError(f"Unknown configuration field: {name}")
        if isinstance(defaults[key], dict):
            result[key] = _merge_config(defaults[key], value, name)
        else:
            result[key] = value
    return result


def _validate_config(config, prefix=""):
    """校验配置的数值类型、范围和首版步长约束。

    Args:
        config: 待校验的配置字典。
        prefix: 当前字段路径。

    Returns:
        None: 校验成功时正常返回。

    Raises:
        ValueError: 字段类型或数值范围不合法。
    """
    integer_fields = {
        "window_size",
        "stride",
        "candidate_batch_size",
        "max_iterations",
        "num_particles",
        "seed",
    }
    for key, value in config.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            _validate_config(value, name)
            continue
        if key == "use_kf":
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be a boolean")
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a finite number")
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
        if key in integer_fields and not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
        allow_zero = (
            key.endswith("radius") or key.startswith("process_") or key == "seed"
        )
        if value < 0 or (value == 0 and not allow_zero):
            raise ValueError(f"{name} is outside its allowed range")
        if key == "stride" and value != 1:
            raise ValueError("Only stride=1 is supported")
        if key == "resample_threshold" and value > 1:
            raise ValueError(f"{name} must be in (0, 1]")


def load_config(default_path, config_path=None):
    """加载默认配置和可选覆盖文件。

    Args:
        default_path: 随库分发的默认 YAML 路径。
        config_path: 可选的完整或局部 YAML 配置路径。

    Returns:
        dict: 经过校验的独立配置。

    Raises:
        ValueError: YAML 结构、字段或数值不合法。
        OSError: 配置文件无法读取。
    """
    with Path(default_path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if config_path is not None:
        with Path(config_path).open(encoding="utf-8") as stream:
            overrides = yaml.safe_load(stream)
        config = _merge_config(config, {} if overrides is None else overrides)
    _validate_config(config)
    return config
