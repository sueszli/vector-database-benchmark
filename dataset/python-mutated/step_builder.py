import os
from enum import Enum
from typing import Dict, List, Optional
from .images.versions import BUILDKITE_TEST_IMAGE_VERSION
from .python_version import AvailablePythonVersion
from .utils import CommandStep, safe_getenv
DEFAULT_TIMEOUT_IN_MIN = 25
DOCKER_PLUGIN = 'docker#v3.7.0'
ECR_PLUGIN = 'ecr#v2.2.0'
AWS_ACCOUNT_ID = os.environ.get('AWS_ACCOUNT_ID')
AWS_ECR_REGION = 'us-west-2'

class BuildkiteQueue(Enum):
    DOCKER = safe_getenv('BUILDKITE_DOCKER_QUEUE')
    MEDIUM = safe_getenv('BUILDKITE_MEDIUM_QUEUE')
    WINDOWS = safe_getenv('BUILDKITE_WINDOWS_QUEUE')

    @classmethod
    def contains(cls, value: object) -> bool:
        if False:
            return 10
        return isinstance(value, cls)

class CommandStepBuilder:
    _step: CommandStep

    def __init__(self, label: str, key: Optional[str]=None, timeout_in_minutes: int=DEFAULT_TIMEOUT_IN_MIN):
        if False:
            return 10
        self._step = {'agents': {'queue': BuildkiteQueue.MEDIUM.value}, 'label': label, 'timeout_in_minutes': timeout_in_minutes, 'retry': {'automatic': [{'exit_status': -1, 'limit': 2}, {'exit_status': 255, 'limit': 2}], 'manual': {'permit_on_passed': True}}}
        if key is not None:
            self._step['key'] = key

    def run(self, *commands: str) -> 'CommandStepBuilder':
        if False:
            while True:
                i = 10
        self._step['commands'] = ['time ' + cmd for cmd in commands]
        return self

    def _base_docker_settings(self) -> Dict[str, object]:
        if False:
            i = 10
            return i + 15
        return {'shell': ['/bin/bash', '-xeuc'], 'always-pull': True, 'mount-ssh-agent': True}

    def on_python_image(self, image: str, env: Optional[List[str]]=None) -> 'CommandStepBuilder':
        if False:
            return 10
        settings = self._base_docker_settings()
        settings['image'] = f'{AWS_ACCOUNT_ID}.dkr.ecr.{AWS_ECR_REGION}.amazonaws.com/{image}'
        settings['volumes'] = ['/var/run/docker.sock:/var/run/docker.sock', '/tmp:/tmp']
        settings['network'] = 'kind'
        buildkite_envvars = [env for env in list(os.environ.keys()) if env.startswith('BUILDKITE') or env.startswith('CI_')]
        settings['environment'] = ['PYTEST_DEBUG_TEMPROOT=/tmp'] + buildkite_envvars + (env or [])
        ecr_settings = {'login': True, 'no-include-email': True, 'account-ids': AWS_ACCOUNT_ID, 'region': AWS_ECR_REGION, 'retries': 2}
        self._step['plugins'] = [{ECR_PLUGIN: ecr_settings}, {DOCKER_PLUGIN: settings}]
        return self

    def on_test_image(self, ver: AvailablePythonVersion, env: Optional[List[str]]=None) -> 'CommandStepBuilder':
        if False:
            while True:
                i = 10
        if not isinstance(ver, AvailablePythonVersion):
            raise Exception(f'Unsupported python version for test image: {ver}.')
        return self.on_python_image(image=f'buildkite-test:py{ver}-{BUILDKITE_TEST_IMAGE_VERSION}', env=env)

    def with_timeout(self, num_minutes: Optional[int]) -> 'CommandStepBuilder':
        if False:
            while True:
                i = 10
        if num_minutes is not None:
            self._step['timeout_in_minutes'] = num_minutes
        return self

    def with_retry(self, num_retries: Optional[int]) -> 'CommandStepBuilder':
        if False:
            for i in range(10):
                print('nop')
        if num_retries is not None:
            self._step['retry'] = {'automatic': {'limit': num_retries}}
        return self

    def with_queue(self, queue: Optional[BuildkiteQueue]) -> 'CommandStepBuilder':
        if False:
            return 10
        if queue is not None:
            assert BuildkiteQueue.contains(queue)
            agents = self._step['agents']
            agents['queue'] = queue.value
        return self

    def with_dependencies(self, step_keys: Optional[List[str]]) -> 'CommandStepBuilder':
        if False:
            print('Hello World!')
        if step_keys is not None:
            self._step['depends_on'] = step_keys
        return self

    def with_skip(self, skip_reason: Optional[str]) -> 'CommandStepBuilder':
        if False:
            print('Hello World!')
        if skip_reason:
            self._step['skip'] = skip_reason
        return self

    def build(self) -> CommandStep:
        if False:
            i = 10
            return i + 15
        return self._step