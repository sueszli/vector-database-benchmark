import os
import sys

def sysAttributes():
    if False:
        return 10
    return (sys.version_info, sys.version_info[0], sys.version_info.major, sys.version, sys.platform, sys.maxsize)

def osAttributes():
    if False:
        for i in range(10):
            print('nop')
    return os.name