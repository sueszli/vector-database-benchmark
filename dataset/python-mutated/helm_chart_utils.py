from __future__ import annotations
import os
from pathlib import Path
import yaml
CHART_DIR = Path(__file__).resolve().parents[2] / 'chart'
CHART_YAML_PATH = os.path.join(CHART_DIR, 'Chart.yaml')

def chart_yaml() -> dict:
    if False:
        while True:
            i = 10
    with open(CHART_YAML_PATH) as f:
        return yaml.safe_load(f)

def chart_version() -> str:
    if False:
        for i in range(10):
            print('nop')
    return chart_yaml()['version']