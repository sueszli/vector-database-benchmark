import logging
import platform
from distutils.version import LooseVersion
from typing import List, Optional
from aiohttp import ClientResponse, ClientSession, ClientTimeout
from ipv8.taskmanager import TaskManager
from tribler.core import notifications
from tribler.core.utilities.notifier import Notifier
from tribler.core.version import version_id
six_hours = 6 * 3600

class VersionCheckManager(TaskManager):
    DEFAULT_URLS = [f'https://release.tribler.org/releases/latest?current={version_id}', 'https://api.github.com/repos/tribler/tribler/releases/latest']

    def __init__(self, notifier: Notifier, check_interval: int=six_hours, request_timeout: int=5, urls: List[str]=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self._logger = logging.getLogger(self.__class__.__name__)
        self.notifier = notifier
        self.check_interval = check_interval
        self.timeout = request_timeout
        self.urls = urls or self.DEFAULT_URLS

    def start(self):
        if False:
            for i in range(10):
                print('nop')
        if 'GIT' not in version_id:
            self.register_task('tribler version check', self._check_urls, interval=self.check_interval, delay=0)

    async def stop(self):
        await self.shutdown_task_manager()

    @property
    def timeout(self):
        if False:
            return 10
        return self._timeout.total

    @timeout.setter
    def timeout(self, value: float):
        if False:
            while True:
                i = 10
        self._timeout = ClientTimeout(total=value)

    async def _check_urls(self) -> Optional[ClientResponse]:
        for version_check_url in self.urls:
            if (result := (await self._request_new_version(version_check_url))):
                return result

    async def _request_new_version(self, version_check_url: str) -> Optional[ClientResponse]:
        try:
            return await self._raw_request_new_version(version_check_url)
        except Exception as e:
            self._logger.warning(e)

    async def _raw_request_new_version(self, version_check_url: str) -> Optional[ClientResponse]:
        headers = {'User-Agent': self._get_user_agent_string(version_id, platform)}
        async with ClientSession(raise_for_status=True) as session:
            response = await session.get(version_check_url, headers=headers, timeout=self.timeout)
            response_dict = await response.json(content_type=None)
            version = response_dict['name'][1:]
            if LooseVersion(version) > LooseVersion(version_id):
                self.notifier[notifications.tribler_new_version](version)
                return response

    @staticmethod
    def _get_user_agent_string(tribler_version, platform_module):
        if False:
            print('Hello World!')
        machine = platform_module.machine()
        os_name = platform_module.system()
        os_release = platform_module.release()
        python_version = platform_module.python_version()
        program_achitecture = platform_module.architecture()[0]
        user_agent = f'Tribler/{tribler_version} (machine={machine}; os={os_name} {os_release}; python={python_version}; executable={program_achitecture})'
        return user_agent