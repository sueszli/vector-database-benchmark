import os
from typing import Optional
import appdirs
APP_NAME = 'golem'
DEFAULT_DATA_DIR = 'default'

def get_local_datadir(name: Optional[str]=None, root_dir: Optional[str]=None, env_suffix: Optional[str]=None) -> str:
    if False:
        return 10
    " Helper function for datadir transition.\n\n        It returns path to a data directory of given name in 'data' dir.\n        Usage should be avoid at all costs. It is always better to ask for\n        a dir the upper layer (like Client instance).\n    "
    if not name:
        name = DEFAULT_DATA_DIR
    if not env_suffix:
        from golem.config.active import DATA_DIR
        env_suffix = DATA_DIR
    if not root_dir:
        root_dir = os.path.join(appdirs.user_data_dir(APP_NAME), name)
    return os.path.join(root_dir, env_suffix)