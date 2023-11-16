"""
Copyright (c) 2014-2023 Maltrail developers (https://github.com/stamparm/maltrail/)
See the file 'LICENSE' for copying permission
"""
import re
from core.common import retrieve_content
__url__ = 'http://vxvault.net/URL_List.php'
__check__ = 'VX Vault'
__info__ = 'malware'
__reference__ = 'vxvault.net'

def fetch():
    if False:
        for i in range(10):
            print('nop')
    retval = {}
    content = retrieve_content(__url__)
    if __check__ in content:
        for line in content.split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '://' in line:
                line = re.search('://(.*)', line).group(1)
                retval[line] = (__info__, __reference__)
    return retval