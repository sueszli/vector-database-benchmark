"""Show battery information.

$ python3 scripts/battery.py
charge:     74%
left:       2:11:31
status:     discharging
plugged in: no
"""
from __future__ import print_function
import sys
import psutil

def secs2hours(secs):
    if False:
        for i in range(10):
            print('nop')
    (mm, ss) = divmod(secs, 60)
    (hh, mm) = divmod(mm, 60)
    return '%d:%02d:%02d' % (hh, mm, ss)

def main():
    if False:
        print('Hello World!')
    if not hasattr(psutil, 'sensors_battery'):
        return sys.exit('platform not supported')
    batt = psutil.sensors_battery()
    if batt is None:
        return sys.exit('no battery is installed')
    print('charge:     %s%%' % round(batt.percent, 2))
    if batt.power_plugged:
        print('status:     %s' % ('charging' if batt.percent < 100 else 'fully charged'))
        print('plugged in: yes')
    else:
        print('left:       %s' % secs2hours(batt.secsleft))
        print('status:     %s' % 'discharging')
        print('plugged in: no')
if __name__ == '__main__':
    main()