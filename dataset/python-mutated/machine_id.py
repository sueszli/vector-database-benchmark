"""
This class is responsible for getting the machine GUID
"""
import subprocess
import sys
import typing

class MachineID:
    """
    This class is responsible for getting the machine GUID
    """

    @staticmethod
    def _run(cmd) -> typing.Optional[str]:
        if False:
            i = 10
            return i + 15
        try:
            return subprocess.run(cmd, shell=True, capture_output=True, check=True, encoding='utf-8').stdout.strip()
        except:
            return None

    @staticmethod
    def get() -> typing.Optional[str]:
        if False:
            i = 10
            return i + 15
        '\n        This function returns the machine UUID\n        :return:    the machine UUID, or None\n        '
        if sys.platform == 'darwin':
            return MachineID._run('ioreg -d2 -c IOPlatformExpertDevice | awk -F\\" \'/IOPlatformUUID/{print $(NF-1)}\'')
        if sys.platform == 'win32' or sys.platform == 'cygwin' or sys.platform == 'msys':
            return MachineID._run('wmic csproduct get uuid').split('\n')[2].strip()
        if sys.platform.startswith('linux'):
            return MachineID._run('cat /var/lib/dbus/machine-id') or MachineID._run('cat /etc/machine-id')
        if sys.platform.startswith('openbsd') or sys.platform.startswith('freebsd'):
            return MachineID._run('cat /etc/hostid') or MachineID._run('kenv -q smbios.system.uuid')
        return None