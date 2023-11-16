from __future__ import annotations
import os
from functools import lru_cache
from black import Mode, TargetVersion, format_str, parse_pyproject_toml
from airflow_breeze.utils.path_utils import AIRFLOW_SOURCES_ROOT

@lru_cache(maxsize=None)
def _black_mode() -> Mode:
    if False:
        i = 10
        return i + 15
    config = parse_pyproject_toml(os.path.join(AIRFLOW_SOURCES_ROOT, 'pyproject.toml'))
    target_versions = {TargetVersion[val.upper()] for val in config.get('target_version', ())}
    return Mode(target_versions=target_versions, line_length=config.get('line_length', Mode.line_length))

def black_format(content) -> str:
    if False:
        return 10
    return format_str(content, mode=_black_mode())