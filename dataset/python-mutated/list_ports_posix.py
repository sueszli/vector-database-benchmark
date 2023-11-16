"""The ``comports`` function is expected to return an iterable that yields tuples
of 3 strings: port name, human readable description and a hardware ID.

As currently no method is known to get the second two strings easily, they are
currently just identical to the port name.
"""
from __future__ import absolute_import
import glob
import sys
import os
from serial.tools import list_ports_common
plat = sys.platform.lower()
if plat[:5] == 'linux':
    from serial.tools.list_ports_linux import comports
elif plat[:6] == 'darwin':
    from serial.tools.list_ports_osx import comports
elif plat == 'cygwin':

    def comports(include_links=False):
        if False:
            i = 10
            return i + 15
        devices = set(glob.glob('/dev/ttyS*'))
        if include_links:
            devices.update(list_ports_common.list_links(devices))
        return [list_ports_common.ListPortInfo(d) for d in devices]
elif plat[:7] == 'openbsd':

    def comports(include_links=False):
        if False:
            print('Hello World!')
        devices = set(glob.glob('/dev/cua*'))
        if include_links:
            devices.update(list_ports_common.list_links(devices))
        return [list_ports_common.ListPortInfo(d) for d in devices]
elif plat[:3] == 'bsd' or plat[:7] == 'freebsd':

    def comports(include_links=False):
        if False:
            while True:
                i = 10
        devices = set(glob.glob('/dev/cua*[!.init][!.lock]'))
        if include_links:
            devices.update(list_ports_common.list_links(devices))
        return [list_ports_common.ListPortInfo(d) for d in devices]
elif plat[:6] == 'netbsd':

    def comports(include_links=False):
        if False:
            for i in range(10):
                print('nop')
        'scan for available ports. return a list of device names.'
        devices = set(glob.glob('/dev/dty*'))
        if include_links:
            devices.update(list_ports_common.list_links(devices))
        return [list_ports_common.ListPortInfo(d) for d in devices]
elif plat[:4] == 'irix':

    def comports(include_links=False):
        if False:
            while True:
                i = 10
        'scan for available ports. return a list of device names.'
        devices = set(glob.glob('/dev/ttyf*'))
        if include_links:
            devices.update(list_ports_common.list_links(devices))
        return [list_ports_common.ListPortInfo(d) for d in devices]
elif plat[:2] == 'hp':

    def comports(include_links=False):
        if False:
            i = 10
            return i + 15
        'scan for available ports. return a list of device names.'
        devices = set(glob.glob('/dev/tty*p0'))
        if include_links:
            devices.update(list_ports_common.list_links(devices))
        return [list_ports_common.ListPortInfo(d) for d in devices]
elif plat[:5] == 'sunos':

    def comports(include_links=False):
        if False:
            i = 10
            return i + 15
        'scan for available ports. return a list of device names.'
        devices = set(glob.glob('/dev/tty*c'))
        if include_links:
            devices.update(list_ports_common.list_links(devices))
        return [list_ports_common.ListPortInfo(d) for d in devices]
elif plat[:3] == 'aix':

    def comports(include_links=False):
        if False:
            while True:
                i = 10
        'scan for available ports. return a list of device names.'
        devices = set(glob.glob('/dev/tty*'))
        if include_links:
            devices.update(list_ports_common.list_links(devices))
        return [list_ports_common.ListPortInfo(d) for d in devices]
else:
    import serial
    sys.stderr.write("don't know how to enumerate ttys on this system.\n! I you know how the serial ports are named send this information to\n! the author of this module:\n\nsys.platform = {!r}\nos.name = {!r}\npySerial version = {}\n\nalso add the naming scheme of the serial ports and with a bit luck you can get\nthis module running...\n".format(sys.platform, os.name, serial.VERSION))
    raise ImportError("Sorry: no implementation for your platform ('{}') available".format(os.name))
if __name__ == '__main__':
    for (port, desc, hwid) in sorted(comports()):
        print('{}: {} [{}]'.format(port, desc, hwid))