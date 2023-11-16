import subprocess
from jadi import component
from aj.plugins.services.api import ServiceManager, Service, ServiceOperationError

@component(ServiceManager)
class SystemdServiceManager(ServiceManager):
    """
    Manager for systemd units.
    """
    id = 'systemd'
    name = 'systemd'

    @classmethod
    def __verify__(cls):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test if systemd is installed.\n\n        :return: Response from which.\n        :rtype: bool\n        '
        return subprocess.call(['which', 'systemctl']) == 0

    def __init__(self, context):
        if False:
            i = 10
            return i + 15
        pass

    def list(self, units=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Generator of all units service in systemd.\n\n        :param units: List of services names\n        :type units: list of strings\n        :return: Service object\n        :rtype: Service\n        '
        if not units:
            units = [x.split()[0] for x in subprocess.check_output(['systemctl', 'list-unit-files', '--no-legend', '--no-pager', '-la']).decode().splitlines() if x]
            units = [x for x in units if x.endswith('.service') and '@.service' not in x]
            units = list(set(units))
        cmd = ['systemctl', 'show', '-o', 'json', '--full', '--all'] + units
        used_names = set()
        unit = {}
        for l in subprocess.check_output(cmd).decode().splitlines() + [None]:
            if not l:
                if len(unit) > 0:
                    svc = Service(self)
                    svc.id = unit['Id']
                    (svc.name, _) = svc.id.rsplit('.', 1)
                    svc.name = svc.name.replace('\\x2d', '-')
                    svc.running = unit['SubState'] == 'running'
                    svc.state = 'running' if svc.running else 'stopped'
                    svc.enabled = unit['UnitFileState'] == 'enabled'
                    svc.static = unit['UnitFileState'] == 'static'
                    if svc.name not in used_names:
                        yield svc
                    used_names.add(svc.name)
                unit = {}
            elif '=' in l:
                (k, v) = l.split('=', 1)
                unit[k] = v

    def get_service(self, _id):
        if False:
            i = 10
            return i + 15
        '\n        Get informations from systemd for one specified service.\n\n        :param _id: Service name\n        :type _id: string\n        :return: Service object\n        :rtype: Service\n        '
        for s in self.list(units=[_id]):
            return s

    def get_status(selfself, _id):
        if False:
            print('Hello World!')
        '\n\n        :param _id: Service name\n        :type _id: string\n        :return: Service status\n        :rtype: string\n        '
        return subprocess.check_output(['systemctl', 'status', _id, '--no-pager']).decode()

    def daemon_reload(self):
        if False:
            print('Hello World!')
        '\n        Basically restart a service.\n        '
        subprocess.check_call(['systemctl', 'daemon-reload'], close_fds=True)

    def start(self, _id):
        if False:
            print('Hello World!')
        '\n        Basically start a service.\n\n        :param _id: Service name\n        :type _id: string\n        '
        try:
            subprocess.check_call(['systemctl', 'start', _id], close_fds=True)
        except subprocess.CalledProcessError as e:
            raise ServiceOperationError(e)

    def stop(self, _id):
        if False:
            for i in range(10):
                print('nop')
        '\n        Basically stop a service.\n\n        :param _id: Service name\n        :type _id: string\n        '
        try:
            subprocess.check_call(['systemctl', 'stop', _id], close_fds=True)
        except subprocess.CalledProcessError as e:
            raise ServiceOperationError(e)

    def restart(self, _id):
        if False:
            return 10
        '\n        Basically restart a service.\n\n        :param _id: Service name\n        :type _id: string\n        '
        try:
            subprocess.check_call(['systemctl', 'restart', _id], close_fds=True)
        except subprocess.CalledProcessError as e:
            raise ServiceOperationError(e)

    def kill(self, _id):
        if False:
            while True:
                i = 10
        '\n        Basically kill a service.\n\n        :param _id: Service name\n        :type _id: string\n        '
        try:
            subprocess.check_call(['systemctl', 'kill -s SIGKILL', _id], close_fds=True)
        except subprocess.CalledProcessError as e:
            raise ServiceOperationError(e)

    def disable(self, _id):
        if False:
            return 10
        '\n        Basically disable a service.\n\n        :param _id: Service name\n        :type _id: string\n        '
        try:
            self.stop(_id)
            subprocess.check_call(['systemctl', 'disable', _id], close_fds=True)
            self.daemon_reload()
        except subprocess.CalledProcessError as e:
            raise ServiceOperationError(e)

    def enable(self, _id):
        if False:
            return 10
        '\n        Basically enable a service.\n\n        :param _id: Service name\n        :type _id: string\n        '
        try:
            subprocess.check_call(['systemctl', 'enable', _id], close_fds=True)
            self.daemon_reload()
        except subprocess.CalledProcessError as e:
            raise ServiceOperationError(e)