import pathlib
from typing import Dict, Optional, Union
from deeplake.util.path import is_hub_cloud_path

def parse_runtime_parameters(path: Union[str, pathlib.Path], runtime: Optional[Dict]=None):
    if False:
        for i in range(10):
            print('nop')
    'Parse runtime parameters from a dictionary.\n    Will become more helpful as clutter in the paramter increases\n\n    Args:\n        path (Union[str, pathlib.Path]): path to the dataset.\n        runtime (Optional[Dict]): A dictionary containing runtime parameters.\n\n    Returns:\n        A dictionary containing parsed runtime parameters.\n\n    Raises:\n        ValueError: If invalid runtime parameters are provided.\n    '
    if isinstance(path, pathlib.Path):
        path = str(path)
    if runtime is None:
        runtime = {}
    db_engine = runtime.get('db_engine', False)
    tensor_db = runtime.get('tensor_db', False) or db_engine
    if tensor_db and (not is_hub_cloud_path(path)):
        raise ValueError(f"Path {path} is not a valid Deep Lake cloud path. runtime = {{'tensor_db': True}} can only be used with datasets stored in the Deep Lake Managed Tensor Database.")
    invalid_keys = set(runtime.keys()) - {'db_engine', 'tensor_db'}
    if len(invalid_keys):
        raise ValueError(f'Invalid runtime parameters: {invalid_keys}.')
    return {'tensor_db': tensor_db}