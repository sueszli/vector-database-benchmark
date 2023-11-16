"""A clone of 'pidof' cmdline utility.

$ pidof python
1140 1138 1136 1134 1133 1129 1127 1125 1121 1120 1119
"""
from __future__ import print_function
import sys
import psutil

def pidof(pgname):
    if False:
        return 10
    pids = []
    for proc in psutil.process_iter(['name', 'cmdline']):
        if proc.info['name'] == pgname or (proc.info['cmdline'] and proc.info['cmdline'][0] == pgname):
            pids.append(str(proc.pid))
    return pids

def main():
    if False:
        i = 10
        return i + 15
    if len(sys.argv) != 2:
        sys.exit('usage: %s pgname' % __file__)
    else:
        pgname = sys.argv[1]
    pids = pidof(pgname)
    if pids:
        print(' '.join(pids))
if __name__ == '__main__':
    main()