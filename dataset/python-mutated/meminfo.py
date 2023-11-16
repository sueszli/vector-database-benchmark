"""Print system memory information.

$ python3 scripts/meminfo.py
MEMORY
------
Total      :    9.7G
Available  :    4.9G
Percent    :    49.0
Used       :    8.2G
Free       :    1.4G
Active     :    5.6G
Inactive   :    2.1G
Buffers    :  341.2M
Cached     :    3.2G

SWAP
----
Total      :      0B
Used       :      0B
Free       :      0B
Percent    :     0.0
Sin        :      0B
Sout       :      0B
"""
import psutil
from psutil._common import bytes2human

def pprint_ntuple(nt):
    if False:
        while True:
            i = 10
    for name in nt._fields:
        value = getattr(nt, name)
        if name != 'percent':
            value = bytes2human(value)
        print('%-10s : %7s' % (name.capitalize(), value))

def main():
    if False:
        i = 10
        return i + 15
    print('MEMORY\n------')
    pprint_ntuple(psutil.virtual_memory())
    print('\nSWAP\n----')
    pprint_ntuple(psutil.swap_memory())
if __name__ == '__main__':
    main()