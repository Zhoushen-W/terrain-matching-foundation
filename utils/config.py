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
        ValueError: 配置包含未知字段。
    """
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
    """按约定的字段类型校验数值范围和首版步长约束。

    Args:
        config: 字段类型符合默认配置约定的配置字典。
        prefix: 当前字段路径。

    Returns:
        None: 校验成功时正常返回。

    Raises:
        ValueError: 数值非有限或超出允许范围。
    """
    for key, value in config.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            _validate_config(value, name)
            continue
        if key == "use_kf":
            continue
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
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
        ValueError: 配置包含未知字段、非有限数值或非法范围。
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
