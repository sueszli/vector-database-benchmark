"""
Reads CLI arguments and performs necessary preparation to be able to run the function
"""
import errno
import json
import logging
import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO, Tuple, Type, cast
from samcli.commands._utils.template import TemplateFailedParsingException, TemplateNotFoundException
from samcli.commands.exceptions import ContainersInitializationException
from samcli.commands.local.cli_common.user_exceptions import DebugContextException, InvokeContextException
from samcli.commands.local.lib.debug_context import DebugContext
from samcli.commands.local.lib.local_lambda import LocalLambdaRunner
from samcli.lib.providers.provider import Function, Stack
from samcli.lib.providers.sam_function_provider import RefreshableSamFunctionProvider, SamFunctionProvider
from samcli.lib.providers.sam_stack_provider import SamLocalStackProvider
from samcli.lib.utils import osutils
from samcli.lib.utils.async_utils import AsyncContext
from samcli.lib.utils.packagetype import ZIP
from samcli.lib.utils.stream_writer import StreamWriter
from samcli.local.docker.exceptions import PortAlreadyInUse
from samcli.local.docker.lambda_image import LambdaImage
from samcli.local.docker.manager import ContainerManager
from samcli.local.lambdafn.runtime import LambdaRuntime, WarmLambdaRuntime
from samcli.local.layers.layer_downloader import LayerDownloader
LOG = logging.getLogger(__name__)

class DockerIsNotReachableException(InvokeContextException):
    """
    Docker is not installed or not running at the moment
    """

class InvalidEnvironmentVariablesFileException(InvokeContextException):
    """
    User provided an environment variables file which couldn't be read by SAM CLI
    """

class NoFunctionIdentifierProvidedException(InvokeContextException):
    """
    If template has more than one function defined and user didn't provide any function logical id
    """

class ContainersInitializationMode(Enum):
    EAGER = 'EAGER'
    LAZY = 'LAZY'

class ContainersMode(Enum):
    WARM = 'WARM'
    COLD = 'COLD'

class InvokeContext:
    """
    Sets up a context to invoke Lambda functions locally by parsing all command line arguments necessary for the
    invoke.

    ``start-api`` command will also use this class to read and parse invoke related CLI arguments and setup the
    necessary context to invoke.

    This class *must* be used inside a `with` statement as follows:

        with InvokeContext(**kwargs) as context:
            context.local_lambda_runner.invoke(...)

    This class sets up some resources that need to be cleaned up after the context object is used.
    """

    def __init__(self, template_file: str, function_identifier: Optional[str]=None, env_vars_file: Optional[str]=None, docker_volume_basedir: Optional[str]=None, docker_network: Optional[str]=None, log_file: Optional[str]=None, skip_pull_image: Optional[bool]=None, debug_ports: Optional[Tuple[int]]=None, debug_args: Optional[str]=None, debugger_path: Optional[str]=None, container_env_vars_file: Optional[str]=None, parameter_overrides: Optional[Dict]=None, layer_cache_basedir: Optional[str]=None, force_image_build: Optional[bool]=None, aws_region: Optional[str]=None, aws_profile: Optional[str]=None, warm_container_initialization_mode: Optional[str]=None, debug_function: Optional[str]=None, shutdown: bool=False, container_host: Optional[str]=None, container_host_interface: Optional[str]=None, invoke_images: Optional[str]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Initialize the context\n\n        Parameters\n        ----------\n        template_file str\n            Name or path to template\n        function_identifier str\n            Identifier of the function to invoke\n        env_vars_file str\n            Path to a file containing values for environment variables\n        docker_volume_basedir str\n            Directory for the Docker volume\n        docker_network str\n            Docker network identifier\n        log_file str\n            Path to a file to send container output to. If the file does not exist, it will be created\n        skip_pull_image bool\n            Should we skip pulling the Docker container image?\n        aws_profile str\n            Name of the profile to fetch AWS credentials from\n        debug_ports tuple(int)\n            Ports to bind the debugger to\n        debug_args str\n            Additional arguments passed to the debugger\n        debugger_path str\n            Path to the directory of the debugger to mount on Docker\n        parameter_overrides dict\n            Values for the template parameters\n        layer_cache_basedir str\n            String representing the path to the layer cache directory\n        force_image_build bool\n            Whether or not to force build the image\n        aws_region str\n            AWS region to use\n        warm_container_initialization_mode str\n            Specifies how SAM cli manages the containers when using start-api or start_lambda.\n            Two modes are available:\n            "EAGER": Containers for every function are loaded at startup and persist between invocations.\n            "LAZY": Containers are only loaded when the function is first invoked and persist for additional invocations\n        debug_function str\n            The Lambda function logicalId that will have the debugging options enabled in case of warm containers\n            option is enabled\n        shutdown bool\n            Optional. If True, perform a SHUTDOWN event when tearing down containers. Default False.\n        container_host string\n            Optional. Host of locally emulated Lambda container\n        container_host_interface string\n            Optional. Interface that Docker host binds ports to\n        invoke_images dict\n            Optional. A dictionary that defines the custom invoke image URI of each function\n        '
        self._template_file = template_file
        self._function_identifier = function_identifier
        self._env_vars_file = env_vars_file
        self._docker_volume_basedir = docker_volume_basedir
        self._docker_network = docker_network
        self._log_file = log_file
        self._skip_pull_image = skip_pull_image
        self._debug_ports = debug_ports
        self._debug_args = debug_args
        self._debugger_path = debugger_path
        self._container_env_vars_file = container_env_vars_file
        self._parameter_overrides = parameter_overrides
        self._global_parameter_overrides: Optional[Dict] = None
        if aws_region:
            self._global_parameter_overrides = {'AWS::Region': aws_region}
        self._layer_cache_basedir = layer_cache_basedir
        self._force_image_build = force_image_build
        self._aws_region = aws_region
        self._aws_profile = aws_profile
        self._shutdown = shutdown
        self._container_host = container_host
        self._container_host_interface = container_host_interface
        self._invoke_images = invoke_images
        self._containers_mode = ContainersMode.COLD
        self._containers_initializing_mode = ContainersInitializationMode.LAZY
        if warm_container_initialization_mode:
            self._containers_mode = ContainersMode.WARM
            self._containers_initializing_mode = ContainersInitializationMode(warm_container_initialization_mode)
        self._debug_function = debug_function
        self._function_provider: SamFunctionProvider = None
        self._stacks: List[Stack] = None
        self._env_vars_value: Optional[Dict] = None
        self._container_env_vars_value: Optional[Dict] = None
        self._log_file_handle: Optional[TextIO] = None
        self._debug_context: Optional[DebugContext] = None
        self._layers_downloader: Optional[LayerDownloader] = None
        self._container_manager: Optional[ContainerManager] = None
        self._lambda_runtimes: Optional[Dict[ContainersMode, LambdaRuntime]] = None
        self._local_lambda_runner: Optional[LocalLambdaRunner] = None

    def __enter__(self) -> 'InvokeContext':
        if False:
            print('Hello World!')
        '\n        Performs some basic checks and returns itself when everything is ready to invoke a Lambda function.\n\n        :returns InvokeContext: Returns this object\n        '
        self._stacks = self._get_stacks()
        _function_providers_class: Dict[ContainersMode, Type[SamFunctionProvider]] = {ContainersMode.WARM: RefreshableSamFunctionProvider, ContainersMode.COLD: SamFunctionProvider}
        _function_providers_args: Dict[ContainersMode, List[Any]] = {ContainersMode.WARM: [self._stacks, self._parameter_overrides, self._global_parameter_overrides], ContainersMode.COLD: [self._stacks]}
        if self._docker_volume_basedir:
            _function_providers_args[self._containers_mode].append(True)
        self._function_provider = _function_providers_class[self._containers_mode](*_function_providers_args[self._containers_mode])
        self._env_vars_value = self._get_env_vars_value(self._env_vars_file)
        self._container_env_vars_value = self._get_env_vars_value(self._container_env_vars_file)
        self._log_file_handle = self._setup_log_file(self._log_file)
        if self._containers_mode == ContainersMode.WARM and self._debug_ports and (not self._debug_function):
            if len(self._function_provider.functions) == 1:
                self._debug_function = list(self._function_provider.functions.keys())[0]
            else:
                LOG.info('Warning: you supplied debugging options but you did not specify the --debug-function option. To specify which function you want to debug, please use the --debug-function <function-name>')
                self._debug_ports = None
        self._debug_context = self._get_debug_context(self._debug_ports, self._debug_args, self._debugger_path, self._container_env_vars_value, self._debug_function)
        self._container_manager = self._get_container_manager(self._docker_network, self._skip_pull_image, self._shutdown)
        if not self._container_manager.is_docker_reachable:
            raise DockerIsNotReachableException('Running AWS SAM projects locally requires Docker. Have you got it installed and running?')
        if self._containers_initializing_mode == ContainersInitializationMode.EAGER:
            self._initialize_all_functions_containers()
        for func in self._function_provider.get_all():
            if func.packagetype == ZIP and func.inlinecode:
                LOG.warning('Warning: Inline code found for function %s. Invocation of inline code is not supported for sam local commands.', func.function_id)
                break
        return self

    def __exit__(self, *args: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Cleanup any necessary opened resources\n        '
        if self._log_file_handle:
            self._log_file_handle.close()
            self._log_file_handle = None
        if self._containers_mode == ContainersMode.WARM:
            self._clean_running_containers_and_related_resources()

    def _initialize_all_functions_containers(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Create and run a container for each available lambda function\n        '
        LOG.info('Initializing the lambda functions containers.')

        def initialize_function_container(function: Function) -> None:
            if False:
                i = 10
                return i + 15
            function_config = self.local_lambda_runner.get_invoke_config(function)
            self.lambda_runtime.run(None, function_config, self._debug_context, self._container_host, self._container_host_interface)
        try:
            async_context = AsyncContext()
            for function in self._function_provider.get_all():
                async_context.add_async_task(initialize_function_container, function)
            async_context.run_async(default_executor=False)
            LOG.info('Containers Initialization is done.')
        except KeyboardInterrupt:
            LOG.debug('Ctrl+C was pressed. Aborting containers initialization')
            self._clean_running_containers_and_related_resources()
            raise
        except PortAlreadyInUse as port_inuse_ex:
            raise port_inuse_ex
        except Exception as ex:
            LOG.error('Lambda functions containers initialization failed because of %s', ex)
            self._clean_running_containers_and_related_resources()
            raise ContainersInitializationException('Lambda functions containers initialization failed') from ex

    def _clean_running_containers_and_related_resources(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Clean the running containers and any other related open resources,\n        it is only used when self.lambda_runtime is a WarmLambdaRuntime\n        '
        cast(WarmLambdaRuntime, self.lambda_runtime).clean_running_containers_and_related_resources()
        cast(RefreshableSamFunctionProvider, self._function_provider).stop_observer()

    @property
    def function_identifier(self) -> str:
        if False:
            while True:
                i = 10
        '\n        Returns identifier of the function to invoke. If no function identifier is provided, this method will return\n        logicalID of the only function from the template\n\n        :return string: Name of the function\n        :raises InvokeContextException: If function identifier is not provided\n        '
        if self._function_identifier:
            return self._function_identifier
        all_functions = list(self._function_provider.get_all())
        if len(all_functions) == 1:
            return all_functions[0].name
        all_function_full_paths = [f.full_path for f in all_functions]
        raise NoFunctionIdentifierProvidedException('You must provide a function logical ID when there are more than one functions in your template. Possible options in your template: {}'.format(all_function_full_paths))

    @property
    def lambda_runtime(self) -> LambdaRuntime:
        if False:
            return 10
        if not self._lambda_runtimes:
            layer_downloader = LayerDownloader(self._layer_cache_basedir, self.get_cwd(), self._stacks)
            image_builder = LambdaImage(layer_downloader, self._skip_pull_image, self._force_image_build, invoke_images=self._invoke_images)
            self._lambda_runtimes = {ContainersMode.WARM: WarmLambdaRuntime(self._container_manager, image_builder), ContainersMode.COLD: LambdaRuntime(self._container_manager, image_builder)}
        return self._lambda_runtimes[self._containers_mode]

    @property
    def local_lambda_runner(self) -> LocalLambdaRunner:
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns an instance of the runner capable of running Lambda functions locally\n\n        :return samcli.commands.local.lib.local_lambda.LocalLambdaRunner: Runner configured to run Lambda functions\n            locally\n        '
        if self._local_lambda_runner:
            return self._local_lambda_runner
        self._local_lambda_runner = LocalLambdaRunner(local_runtime=self.lambda_runtime, function_provider=self._function_provider, cwd=self.get_cwd(), aws_profile=self._aws_profile, aws_region=self._aws_region, env_vars_values=self._env_vars_value, debug_context=self._debug_context, container_host=self._container_host, container_host_interface=self._container_host_interface)
        return self._local_lambda_runner

    @property
    def stdout(self) -> StreamWriter:
        if False:
            i = 10
            return i + 15
        '\n        Returns stream writer for stdout to output Lambda function logs to\n\n        Returns\n        -------\n        samcli.lib.utils.stream_writer.StreamWriter\n            Stream writer for stdout\n        '
        stream = self._log_file_handle if self._log_file_handle else osutils.stdout()
        return StreamWriter(stream, auto_flush=True)

    @property
    def stderr(self) -> StreamWriter:
        if False:
            return 10
        '\n        Returns stream writer for stderr to output Lambda function errors to\n\n        Returns\n        -------\n        samcli.lib.utils.stream_writer.StreamWriter\n            Stream writer for stderr\n        '
        stream = self._log_file_handle if self._log_file_handle else osutils.stderr()
        return StreamWriter(stream, auto_flush=True)

    @property
    def stacks(self) -> List[Stack]:
        if False:
            i = 10
            return i + 15
        '\n        Returns the list of stacks (including the root stack and all children stacks)\n\n        :return list: list of stacks\n        '
        return self._function_provider.stacks

    def get_cwd(self) -> str:
        if False:
            print('Hello World!')
        '\n        Get the working directory. This is usually relative to the directory that contains the template. If a Docker\n        volume location is specified, it takes preference\n\n        All Lambda function code paths are resolved relative to this working directory\n\n        :return string: Working directory\n        '
        cwd = os.path.dirname(os.path.abspath(self._template_file))
        if self._docker_volume_basedir:
            cwd = self._docker_volume_basedir
        return cwd

    @property
    def _is_debugging(self) -> bool:
        if False:
            return 10
        return bool(self._debug_context)

    def _get_stacks(self) -> List[Stack]:
        if False:
            return 10
        try:
            (stacks, _) = SamLocalStackProvider.get_stacks(self._template_file, parameter_overrides=self._parameter_overrides, global_parameter_overrides=self._global_parameter_overrides)
            return stacks
        except (TemplateNotFoundException, TemplateFailedParsingException) as ex:
            LOG.debug("Can't read stacks information, either template is not found or it is invalid", exc_info=ex)
            raise ex

    @staticmethod
    def _get_env_vars_value(filename: Optional[str]) -> Optional[Dict]:
        if False:
            i = 10
            return i + 15
        '\n        If the user provided a file containing values of environment variables, this method will read the file and\n        return its value\n\n        :param string filename: Path to file containing environment variable values\n        :return dict: Value of environment variables, if provided. None otherwise\n        :raises InvokeContextException: If the file was not found or not a valid JSON\n        '
        if not filename:
            return None
        try:
            with open(filename, 'r') as fp:
                return cast(Dict, json.load(fp))
        except Exception as ex:
            raise InvalidEnvironmentVariablesFileException('Could not read environment variables overrides from file {}: {}'.format(filename, str(ex))) from ex

    @staticmethod
    def _setup_log_file(log_file: Optional[str]) -> Optional[TextIO]:
        if False:
            return 10
        '\n        Open a log file if necessary and return the file handle. This will create a file if it does not exist\n\n        :param string log_file: Path to a file where the logs should be written to\n        :return: Handle to the opened log file, if necessary. None otherwise\n        '
        if not log_file:
            return None
        return open(log_file, 'w', encoding='utf8')

    @staticmethod
    def _get_debug_context(debug_ports: Optional[Tuple[int]], debug_args: Optional[str], debugger_path: Optional[str], container_env_vars: Optional[Dict[str, str]], debug_function: Optional[str]=None) -> DebugContext:
        if False:
            i = 10
            return i + 15
        '\n        Creates a DebugContext if the InvokeContext is in a debugging mode\n\n        Parameters\n        ----------\n        debug_ports tuple(int)\n             Ports to bind the debugger to\n        debug_args str\n            Additional arguments passed to the debugger\n        debugger_path str\n            Path to the directory of the debugger to mount on Docker\n        container_env_vars dict\n            Dictionary containing debugging based environmental variables.\n        debug_function str\n            The Lambda function logicalId that will have the debugging options enabled in case of warm containers\n            option is enabled\n\n        Returns\n        -------\n        samcli.commands.local.lib.debug_context.DebugContext\n            Object representing the DebugContext\n\n        Raises\n        ------\n        samcli.commands.local.cli_common.user_exceptions.DebugContext\n            When the debugger_path is not valid\n        '
        if debug_ports and debugger_path:
            try:
                debugger = Path(debugger_path).resolve(strict=True)
            except OSError as error:
                if error.errno == errno.ENOENT:
                    raise DebugContextException("'{}' could not be found.".format(debugger_path)) from error
                raise error
            if not debugger.is_dir():
                raise DebugContextException("'{}' should be a directory with the debugger in it.".format(debugger_path))
            debugger_path = str(debugger)
        return DebugContext(debug_ports=debug_ports, debug_args=debug_args, debugger_path=debugger_path, debug_function=debug_function, container_env_vars=container_env_vars)

    @staticmethod
    def _get_container_manager(docker_network: Optional[str], skip_pull_image: Optional[bool], shutdown: Optional[bool]) -> ContainerManager:
        if False:
            for i in range(10):
                print('nop')
        '\n        Creates a ContainerManager with specified options\n\n        Parameters\n        ----------\n        docker_network str\n            Docker network identifier\n        skip_pull_image bool\n            Should the manager skip pulling the image\n        shutdown bool\n            Should SHUTDOWN events be sent when tearing down image\n\n        Returns\n        -------\n        samcli.local.docker.manager.ContainerManager\n            Object representing Docker container manager\n        '
        return ContainerManager(docker_network_id=docker_network, skip_pull_image=skip_pull_image, do_shutdown_event=shutdown)