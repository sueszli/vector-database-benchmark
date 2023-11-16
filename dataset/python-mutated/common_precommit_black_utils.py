from __future__ import annotations
import os
import sys
from functools import lru_cache
from pathlib import Path
from black import Mode, TargetVersion, format_str, parse_pyproject_toml
sys.path.insert(0, str(Path(__file__).parent.resolve()))
from common_precommit_utils import AIRFLOW_BREEZE_SOURCES_PATH

@lru_cache(maxsize=None)
def black_mode(is_pyi: bool=Mode.is_pyi) -> Mode:
    if False:
        while True:
            i = 10
    config = parse_pyproject_toml(os.fspath(AIRFLOW_BREEZE_SOURCES_PATH / 'pyproject.toml'))
    target_versions = {TargetVersion[val.upper()] for val in config.get('target_version', ())}
    return Mode(target_versions=target_versions, line_length=config.get('line_length', Mode.line_length), is_pyi=is_pyi)

def black_format(content: str, is_pyi: bool=Mode.is_pyi) -> str:
    if False:
        print('Hello World!')
    return format_str(content, mode=black_mode(is_pyi=is_pyi))