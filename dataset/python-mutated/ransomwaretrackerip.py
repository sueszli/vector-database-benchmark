"""
Copyright (c) 2014-2023 Maltrail developers (https://github.com/stamparm/maltrail/)
See the file 'LICENSE' for copying permission
"""
from core.common import retrieve_content
__url__ = 'https://ransomwaretracker.abuse.ch/downloads/RW_IPBL.txt'
__check__ = 'questions'
__info__ = 'ransomware (malware)'
__reference__ = 'abuse.ch'

def fetch():
    if False:
        return 10
    retval = {}
    content = retrieve_content(__url__)
    if __check__ in content:
        for line in content.split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            retval[line] = (__info__, __reference__)
    return retval