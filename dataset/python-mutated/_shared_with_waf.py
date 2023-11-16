"""
Code to be shared by PyInstaller and the bootloader/wscript file.

This code must not assume that either PyInstaller or any of its dependencies installed. I.e., the only imports allowed
in here are standard library ones. Within reason, it is preferable that this file should still run under Python 2.7 as
many compiler docker images still have only Python 2 installed.
"""
import platform
import re

def _pyi_machine(machine, system):
    if False:
        return 10
    "\n    Choose an intentionally simplified architecture identifier to be used in the bootloader's directory name.\n\n    Args:\n        machine:\n            The output of ``platform.machine()`` or any known architecture alias or shorthand that may be used by a\n            C compiler.\n        system:\n            The output of ``platform.system()`` on the target machine.\n    Returns:\n        Either a string tag or, on platforms that don't need an architecture tag, ``None``.\n\n    Ideally, we would just use ``platform.machine()`` directly, but that makes cross-compiling the bootloader almost\n    impossible, because you need to know at compile time exactly what ``platform.machine()`` will be at run time, based\n    only on the machine name alias or shorthand reported by the C compiler at the build time. Rather, use a loose\n    differentiation, and trust that anyone mixing armv6l with armv6h knows what they are doing.\n    "
    if platform.machine() == 'sw_64' or platform.machine() == 'loongarch64':
        return platform.machine()
    if system == 'Windows':
        if machine.lower().startswith('arm'):
            return 'arm'
        else:
            return 'intel'
    if system != 'Linux':
        return
    if machine.startswith(('arm', 'aarch')):
        return 'arm'
    if machine in 'thumb':
        return 'arm'
    if machine in ('x86_64', 'x64', 'x86'):
        return 'intel'
    if re.fullmatch('i[1-6]86', machine):
        return 'intel'
    if machine.startswith(('ppc', 'powerpc')):
        return 'ppc'
    if machine in ('mips64', 'mips'):
        return 'mips'
    if machine.startswith('riscv'):
        return 'riscv'
    if machine in ('s390x',):
        return machine
    return 'unknown'