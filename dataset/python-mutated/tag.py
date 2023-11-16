from deeplake.util.exceptions import InvalidHubPathException
from typing import Tuple
import deeplake

def process_hub_path(path: str) -> Tuple[str, str, str, str]:
    if False:
        return 10
    'Checks whether path is a valid Deep Lake cloud path.'
    tag = path[6:]
    s = tag.split('/')
    if len(s) < 2:
        raise InvalidHubPathException(path)
    path = f'hub://{s[0]}/{s[1]}'
    if len(s) == 3 and s[1] == 'queries' and (not s[2].startswith('.')):
        subdir = f'.queries/{s[2]}'
    else:
        subdir = '/'.join(s[2:])
        if len(s) > 2:
            if not (len(s) == 4 and s[2] == '.queries') and (not deeplake.constants._ENABLE_HUB_SUB_DATASETS):
                raise InvalidHubPathException(path)
    return (path, *s[:2], subdir)