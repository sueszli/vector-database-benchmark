import os
import appdirs
config_dir = appdirs.user_config_dir('Open Interpreter')

def get_storage_path(subdirectory=None):
    if False:
        i = 10
        return i + 15
    if subdirectory is None:
        return config_dir
    else:
        return os.path.join(config_dir, subdirectory)