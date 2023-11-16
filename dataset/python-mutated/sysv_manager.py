import os
import subprocess
from jadi import component
from aj.plugins.services.api import ServiceManager, Service
INIT_D = '/etc/init.d'

@component(ServiceManager)
class SysVServiceManager(ServiceManager):
    """
    Manager for sysv init scripts.
    """
    id = 'sysv'
    name = 'System V'

    @classmethod
    def __verify__(cls):
        if False:
            while True:
                i = 10
        '\n        Test if sysv init scripts are used\n\n        :return: Existence of /etc/init.d\n        :rtype: bool\n        '
        return os.path.exists(INIT_D)

    def __init__(self, context):
        if False:
            while True:
                i = 10
        pass

    def list(self):
        if False:
            i = 10
            return i + 15
        '\n        Generator of all scripts under /etc/init.d.\n\n        :return: Service object\n        :rtype: Service\n        '
        for _id in os.listdir(INIT_D):
            path = os.path.join(INIT_D, _id)
            if _id.startswith('.'):
                continue
            if _id.startswith('rc'):
                continue
            if os.path.islink(path):
                continue
            if os.path.exists(f'/etc/init/{_id}.conf'):
                continue
            yield self.get_service(_id)

    def get_service(self, _id):
        if False:
            i = 10
            return i + 15
        '\n        Get status for one specified init script.\n\n        :param _id: Script name\n        :type _id: string\n        :return: Service object\n        :rtype: Service\n        '
        svc = Service(self)
        svc.id = svc.name = _id
        svc.enabled = True
        svc.static = False
        try:
            svc.running = self._run_action(_id, 'status')
            svc.state = 'running' if svc.running else 'stopped'
        except Exception as e:
            svc.running = False
        return svc

    def _run_action(self, _id, action):
        if False:
            print('Hello World!')
        '\n        Wrapper for basic scripts actions ( restart, start, stop, status ).\n\n        :param _id: Script name\n        :type _id: string\n        :param action: Action ( restart, start, stop, status )\n        :type action: string\n        '
        return subprocess.call([os.path.join(INIT_D, _id), action], close_fds=True) == 0

    def start(self, _id):
        if False:
            for i in range(10):
                print('nop')
        '\n        Basically start a script.\n\n        :param _id: Script name\n        :type _id: string\n        '
        self._run_action(_id, 'start')

    def stop(self, _id):
        if False:
            i = 10
            return i + 15
        '\n        Basically stop a script.\n\n        :param _id: Script name\n        :type _id: string\n        '
        self._run_action(_id, 'stop')

    def restart(self, _id):
        if False:
            return 10
        '\n        Basically restart a script.\n\n        :param _id: Script name\n        :type _id: string\n        '
        self._run_action(_id, 'restart')