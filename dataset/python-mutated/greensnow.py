"""
Copyright (c) 2014-2023 Maltrail developers (https://github.com/stamparm/maltrail/)
See the file 'LICENSE' for copying permission
"""
from core.common import retrieve_content
from core.settings import NAME
__url__ = 'https://blocklist.greensnow.co/greensnow.txt'
__check__ = '.1'
__info__ = 'known attacker'
__reference__ = 'greensnow.co'

def fetch():
    if False:
        while True:
            i = 10
    retval = {}
    content = retrieve_content(__url__, headers={'User-agent': NAME})
    if __check__ in content:
        for line in content.split('\n'):
            line = line.strip()
            if not line or line.startswith('#') or '.' not in line:
                continue
            retval[line] = (__info__, __reference__)
    return retval