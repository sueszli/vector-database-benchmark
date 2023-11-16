import logging
from abc import ABC, abstractmethod
from typing import Type
from plugin import PluginManager
from localstack import config
from localstack.services.lambda_.invocation.lambda_models import FunctionVersion, InvocationResult
from localstack.services.lambda_.invocation.plugins import RuntimeExecutorPlugin
LOG = logging.getLogger(__name__)

class RuntimeExecutor(ABC):
    id: str
    function_version: FunctionVersion

    def __init__(self, id: str, function_version: FunctionVersion) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Runtime executor class responsible for executing a runtime in specific environment\n\n        :param id: ID string of the runtime executor\n        :param function_version: Function version to be executed\n        '
        self.id = id
        self.function_version = function_version

    @abstractmethod
    def start(self, env_vars: dict[str, str]) -> None:
        if False:
            while True:
                i = 10
        '\n        Start the runtime executor with the given environment variables\n\n        :param env_vars:\n        '
        pass

    @abstractmethod
    def stop(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Stop the runtime executor\n        '
        pass

    @abstractmethod
    def get_address(self) -> str:
        if False:
            i = 10
            return i + 15
        '\n        Get the address the runtime executor is available at for the LocalStack container.\n\n        :return: IP address or hostname of the execution environment\n        '
        pass

    @abstractmethod
    def get_endpoint_from_executor(self) -> str:
        if False:
            return 10
        '\n        Get the address of LocalStack the runtime execution environment can communicate with LocalStack\n\n        :return: IP address or hostname of LocalStack (from the view of the execution environment)\n        '
        pass

    @abstractmethod
    def get_runtime_endpoint(self) -> str:
        if False:
            while True:
                i = 10
        '\n        Gets the callback url of our executor endpoint\n\n        :return: Base url of the callback, e.g. "http://123.123.123.123:4566/_localstack_lambda/ID1234" without trailing slash\n        '
        pass

    @abstractmethod
    def invoke(self, payload: dict[str, str]) -> InvocationResult:
        if False:
            i = 10
            return i + 15
        '\n        Send an invocation to the execution environment\n\n        :param payload: Invocation payload\n        '
        pass

    @abstractmethod
    def get_logs(self) -> str:
        if False:
            i = 10
            return i + 15
        'Get all logs of a given execution environment'
        pass

    @classmethod
    @abstractmethod
    def prepare_version(cls, function_version: FunctionVersion) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Prepare a given function version to be executed.\n        Includes all the preparation work necessary for execution, short of starting anything\n\n        :param function_version: Function version to prepare\n        '
        pass

    @classmethod
    @abstractmethod
    def cleanup_version(cls, function_version: FunctionVersion):
        if False:
            i = 10
            return i + 15
        '\n        Cleanup the version preparation for the given version.\n        Should cleanup preparation steps taken by prepare_version\n        :param function_version:\n        '
        pass

    @classmethod
    def validate_environment(cls) -> bool:
        if False:
            print('Hello World!')
        'Validates the setup of the environment and provides an opportunity to log warnings.\n        Returns False if an invalid environment is detected and True otherwise.'
        return True

class LambdaRuntimeException(Exception):

    def __init__(self, message: str):
        if False:
            i = 10
            return i + 15
        super().__init__(message)
EXECUTOR_PLUGIN_MANAGER: PluginManager[Type[RuntimeExecutor]] = PluginManager(RuntimeExecutorPlugin.namespace)

def get_runtime_executor() -> Type[RuntimeExecutor]:
    if False:
        i = 10
        return i + 15
    plugin_name = config.LAMBDA_RUNTIME_EXECUTOR or 'docker'
    if not EXECUTOR_PLUGIN_MANAGER.exists(plugin_name):
        LOG.warning('Invalid specified plugin name %s. Falling back to "docker" runtime executor', plugin_name)
        plugin_name = 'docker'
    return EXECUTOR_PLUGIN_MANAGER.load(plugin_name).load()