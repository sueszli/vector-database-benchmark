import os
import yaml
from dagster._core.telemetry import get_or_create_dir_from_dagster_home
NUX_FILE_STR = 'nux.yaml'

def nux_seen_filepath():
    if False:
        i = 10
        return i + 15
    return os.path.join(get_or_create_dir_from_dagster_home('.nux'), 'nux.yaml')

def set_nux_seen():
    if False:
        while True:
            i = 10
    try:
        with open(nux_seen_filepath(), 'w', encoding='utf8') as nux_seen_file:
            yaml.dump({'seen': 1}, nux_seen_file, default_flow_style=False)
    except Exception:
        return '<<unable_to_write_nux_seen>>'

def get_has_seen_nux():
    if False:
        for i in range(10):
            print('nop')
    try:
        return os.path.exists(nux_seen_filepath())
    except:
        return True