"""NIOS plugin for integration tests."""
from __future__ import annotations
import os
from ....config import IntegrationConfig
from ....containers import run_support_container
from . import CloudEnvironment, CloudEnvironmentConfig, CloudProvider

class NiosProvider(CloudProvider):
    """Nios plugin. Sets up NIOS mock server for tests."""
    DOCKER_IMAGE = 'quay.io/ansible/nios-test-container:2.0.0'

    def __init__(self, args: IntegrationConfig) -> None:
        if False:
            return 10
        super().__init__(args)
        self.__container_from_env = os.environ.get('ANSIBLE_NIOSSIM_CONTAINER')
        '\n        Overrides target container, might be used for development.\n\n        Use ANSIBLE_NIOSSIM_CONTAINER=whatever_you_want if you want\n        to use other image. Omit/empty otherwise.\n        '
        self.image = self.__container_from_env or self.DOCKER_IMAGE
        self.uses_docker = True

    def setup(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Setup cloud resource before delegation and reg cleanup callback.'
        super().setup()
        if self._use_static_config():
            self._setup_static()
        else:
            self._setup_dynamic()

    def _setup_dynamic(self) -> None:
        if False:
            print('Hello World!')
        'Spawn a NIOS simulator within docker container.'
        nios_port = 443
        ports = [nios_port]
        descriptor = run_support_container(self.args, self.platform, self.image, 'nios-simulator', ports)
        if not descriptor:
            return
        self._set_cloud_config('NIOS_HOST', descriptor.name)

    def _setup_static(self) -> None:
        if False:
            print('Hello World!')
        raise NotImplementedError()

class NiosEnvironment(CloudEnvironment):
    """NIOS environment plugin. Updates integration test environment after delegation."""

    def get_environment_config(self) -> CloudEnvironmentConfig:
        if False:
            i = 10
            return i + 15
        'Return environment configuration for use in the test environment after delegation.'
        ansible_vars = dict(nios_provider=dict(host=self._get_cloud_config('NIOS_HOST'), username='admin', password='infoblox'))
        return CloudEnvironmentConfig(ansible_vars=ansible_vars)