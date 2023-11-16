import os
import shutil
from importlib import resources
import yaml
from .local_storage_path import get_storage_path
config_filename = 'config.yaml'
user_config_path = os.path.join(get_storage_path(), config_filename)

def get_config_path(path=user_config_path):
    if False:
        i = 10
        return i + 15
    if not os.path.exists(path):
        if os.path.exists(os.path.join(get_storage_path(), path)):
            path = os.path.join(get_storage_path(), path)
        elif os.path.exists(os.path.join(os.getcwd(), path)):
            path = os.path.join(os.path.curdir, path)
        else:
            if os.path.dirname(path) and (not os.path.exists(os.path.dirname(path))):
                os.makedirs(os.path.dirname(path), exist_ok=True)
            else:
                os.makedirs(get_storage_path(), exist_ok=True)
                path = os.path.join(get_storage_path(), path)
            here = os.path.abspath(os.path.dirname(__file__))
            parent_dir = os.path.dirname(here)
            default_config_path = os.path.join(parent_dir, 'config.yaml')
            new_file = shutil.copy(default_config_path, path)
    return path

def get_config(path=user_config_path):
    if False:
        return 10
    path = get_config_path(path)
    with open(path, 'r') as file:
        return yaml.safe_load(file)