"""
Copyright (c) 2014-2023 Maltrail developers (https://github.com/stamparm/maltrail/)
See the file 'LICENSE' for copying permission
"""
import re
from core.common import retrieve_content
__url__ = 'https://rules.emergingthreats.net/open/suricata/rules/botcc.rules'
__check__ = 'CnC Server'
__info__ = 'potential malware site'
__reference__ = 'emergingthreats.net'

def fetch():
    if False:
        while True:
            i = 10
    retval = {}
    content = retrieve_content(__url__)
    if __check__ in content:
        for match in re.finditer('\\d+\\.\\d+\\.\\d+\\.\\d+', content):
            retval[match.group(0)] = (__info__, __reference__)
    return retval