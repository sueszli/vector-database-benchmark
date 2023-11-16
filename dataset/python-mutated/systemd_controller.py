import logging
from shutil import which
from subprocess import check_output
logger = logging.getLogger()

def systemctl_run(*args):
    if False:
        i = 10
        return i + 15
    try:
        return check_output(['systemctl', '--user', *args]).decode('utf-8').rstrip()
    except Exception:
        return False

class SystemdController:

    def __init__(self, unit: str):
        if False:
            print('Hello World!')
        self.unit = unit

    def can_start(self):
        if False:
            i = 10
            return i + 15
        '\n        :returns: True if unit exists and can start\n        '
        if not which('systemctl'):
            logger.warning('systemctl command missing')
            return False
        status = systemctl_run('show', self.unit)
        if 'NeedDaemonReload=yes' in status:
            logger.info('Reloading systemd daemon')
            systemctl_run('daemon-reload')
            status = systemctl_run('show', self.unit)
        return 'CanStart=yes' in status

    def is_active(self):
        if False:
            i = 10
            return i + 15
        '\n        :returns: True if unit is currently running\n        '
        return systemctl_run('is-active', self.unit) == 'active'

    def is_enabled(self):
        if False:
            return 10
        '\n        :returns: True if unit is set to start automatically\n        '
        return systemctl_run('is-enabled', self.unit) == 'enabled'

    def restart(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        :returns: Restart the service\n        '
        return systemctl_run('restart', self.unit)

    def toggle(self, status):
        if False:
            print('Hello World!')
        '\n        Enable or disable unit\n\n        :param bool status:\n        '
        if not self.can_start():
            msg = 'Autostart is not allowed'
            raise OSError(msg)
        systemctl_run('reenable' if status else 'disable', self.unit)