import os
from typing import List, Optional

def print_warning(*lines: str) -> None:
    if False:
        return 10
    print('**************************************************')
    for line in lines:
        print('*** WARNING: %s' % line)
    print('**************************************************')

def get_path(key: str) -> List[str]:
    if False:
        i = 10
        return i + 15
    return os.environ.get(key, '').split(os.pathsep)

def search_on_path(filenames: List[str]) -> Optional[str]:
    if False:
        for i in range(10):
            print('nop')
    for p in get_path('PATH'):
        for filename in filenames:
            full = os.path.join(p, filename)
            if os.path.exists(full):
                return os.path.abspath(full)
    return None