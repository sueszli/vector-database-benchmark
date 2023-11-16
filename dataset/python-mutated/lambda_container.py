"""
Represents Lambda runtime containers.
"""
import logging
from typing import List
from samcli.lib.utils.packagetype import IMAGE
from samcli.local.docker.lambda_debug_settings import LambdaDebugSettings
from .container import Container
from .lambda_image import LambdaImage, Runtime
LOG = logging.getLogger(__name__)

class LambdaContainer(Container):
    """
    Represents a Lambda runtime container. This class knows how to setup entry points, environment variables,
    exposed ports etc specific to Lambda runtime container. The container management functionality (create/start/stop)
    is provided by the base class
    """
    _WORKING_DIR = '/var/task'
    _DEFAULT_ENTRYPOINT = ['/var/rapid/aws-lambda-rie', '--log-level', 'error']
    _DEBUGGER_VOLUME_MOUNT_PATH = '/tmp/lambci_debug_files'
    _DEFAULT_CONTAINER_DBG_GO_PATH = _DEBUGGER_VOLUME_MOUNT_PATH + '/dlv'
    _DEBUG_ENTRYPOINT_OPTIONS = {'delvePath': _DEFAULT_CONTAINER_DBG_GO_PATH}
    _DEBUGGER_VOLUME_MOUNT = {'bind': _DEBUGGER_VOLUME_MOUNT_PATH, 'mode': 'ro'}

    def __init__(self, runtime, imageuri, handler, packagetype, image_config, code_dir, layers, lambda_image, architecture, memory_mb=128, env_vars=None, debug_options=None, container_host=None, container_host_interface=None, function_full_path=None):
        if False:
            i = 10
            return i + 15
        '\n        Initializes the class\n\n        Parameters\n        ----------\n        runtime str\n            Name of the Lambda runtime\n        imageuri str\n            Name of the Lambda Image which is of the form {image}:{tag}\n        handler str\n            Handler of the function to run\n        packagetype str\n            Package type for the lambda function which is either zip or image.\n        image_config dict\n            Image configuration which can be used set to entrypoint, command and working dir for the container.\n        code_dir str\n            Directory where the Lambda function code is present. This directory will be mounted\n            to the container to execute\n        layers list(str)\n            List of layers\n        lambda_image samcli.local.docker.lambda_image.LambdaImage\n            LambdaImage that can be used to build the image needed for starting the container\n        architecture str\n            Architecture type either x86_64 or arm64 on AWS lambda\n        memory_mb int\n            Optional. Max limit of memory in MegaBytes this Lambda function can use.\n        env_vars dict\n            Optional. Dictionary containing environment variables passed to container\n        debug_options DebugContext\n            Optional. Contains container debugging info (port, debugger path)\n        container_host string\n            Optional. Host of locally emulated Lambda container\n        container_host_interface\n            Optional. Interface that Docker host binds ports to\n        function_full_path str\n            Optional. The function full path, unique in all stacks\n        '
        if not Runtime.has_value(runtime) and (not packagetype == IMAGE):
            raise ValueError('Unsupported Lambda runtime {}'.format(runtime))
        image = LambdaContainer._get_image(lambda_image, runtime, packagetype, imageuri, layers, architecture, function_full_path)
        ports = LambdaContainer._get_exposed_ports(debug_options)
        config = LambdaContainer._get_config(lambda_image, image)
        (entry, container_env_vars) = LambdaContainer._get_debug_settings(runtime, debug_options)
        additional_options = LambdaContainer._get_additional_options(runtime, debug_options)
        additional_volumes = LambdaContainer._get_additional_volumes(runtime, debug_options)
        _work_dir = self._WORKING_DIR
        _entrypoint = None
        _command = None
        if not env_vars:
            env_vars = {}
        if packagetype == IMAGE:
            _command = (image_config.get('Command') if image_config else None) or config.get('Cmd')
            if not env_vars.get('AWS_LAMBDA_FUNCTION_HANDLER', None):
                env_vars['AWS_LAMBDA_FUNCTION_HANDLER'] = _command[0] if isinstance(_command, list) else None
            _additional_entrypoint_args = (image_config.get('EntryPoint') if image_config else None) or config.get('Entrypoint')
            _entrypoint = entry or self._DEFAULT_ENTRYPOINT
            if isinstance(_additional_entrypoint_args, list) and entry == self._DEFAULT_ENTRYPOINT:
                _entrypoint = _entrypoint + _additional_entrypoint_args
            _work_dir = (image_config.get('WorkingDirectory') if image_config else None) or config.get('WorkingDir')
        env_vars = {**env_vars, **container_env_vars}
        super().__init__(image, _command if _command else [], _work_dir, code_dir, memory_limit_mb=memory_mb, exposed_ports=ports, entrypoint=_entrypoint if _entrypoint else entry, env_vars=env_vars, container_opts=additional_options, additional_volumes=additional_volumes, container_host=container_host, container_host_interface=container_host_interface)

    @staticmethod
    def _get_exposed_ports(debug_options):
        if False:
            i = 10
            return i + 15
        '\n        Return Docker container port binding information. If a debug port tuple is given, then we will ask Docker to\n        bind every given port to same port both inside and outside the container ie.\n        Runtime process is started in debug mode with at given port inside the container\n        and exposed to the host machine at the same port.\n\n        :param DebugContext debug_options: Debugging options for the function (includes debug port, args, and path)\n        :return dict: Dictionary containing port binding information. None, if debug_port was not given\n        '
        if not debug_options:
            return None
        if not debug_options.debug_ports:
            return None
        ports_map = {}
        for port in debug_options.debug_ports:
            ports_map[port] = port
        return ports_map

    @staticmethod
    def _get_additional_options(runtime: str, debug_options):
        if False:
            return 10
        '\n        Return additional Docker container options. Used by container debug mode to enable certain container\n        security options.\n        :param runtime: The runtime string\n        :param DebugContext debug_options: DebugContext for the runtime of the container.\n        :return dict: Dictionary containing additional arguments to be passed to container creation.\n        '
        if not debug_options:
            return None
        opts = {}
        if runtime == Runtime.go1x.value:
            opts['security_opt'] = ['seccomp:unconfined']
            opts['cap_add'] = ['SYS_PTRACE']
        return opts

    @staticmethod
    def _get_additional_volumes(runtime, debug_options):
        if False:
            print('Hello World!')
        '\n        Return additional volumes to be mounted in the Docker container. Used by container debug for mapping\n        debugger executable into the container.\n        :param runtime: the runtime string\n        :param DebugContext debug_options: DebugContext for the runtime of the container.\n        :return dict: Dictionary containing volume map passed to container creation.\n        '
        volumes = {}
        if debug_options and debug_options.debugger_path:
            volumes[debug_options.debugger_path] = LambdaContainer._DEBUGGER_VOLUME_MOUNT
        return volumes

    @staticmethod
    def _get_image(lambda_image: LambdaImage, runtime: str, packagetype: str, image: str, layers: List[str], architecture: str, function_name: str):
        if False:
            print('Hello World!')
        '\n\n        Parameters\n        ----------\n        lambda_image : LambdaImage\n            LambdaImage that can be used to build the image needed for starting the container\n        runtime : str\n            Name of the Lambda runtime\n        packagetype : str\n            Package type for the lambda function which is either zip or image.\n        image : str\n            Name of the Lambda Image which is of the form {image}:{tag}\n        layers : List[str]\n            List of layers\n        architecture : str\n            Architecture type either x86_64 or arm64 on AWS lambda\n        function_name: str\n            The name of the lambda function that the container is to invoke\n\n        Returns\n        -------\n        str\n            Name of Docker Image for the given runtime\n        '
        return lambda_image.build(runtime, packagetype, image, layers, architecture, function_name=function_name)

    @staticmethod
    def _get_config(lambda_image, image):
        if False:
            print('Hello World!')
        return lambda_image.get_config(image)

    @staticmethod
    def _get_debug_settings(runtime, debug_options=None):
        if False:
            return 10
        '\n        Returns the entry point for the container. The default value for the entry point is already configured in the\n        Dockerfile. We override this default specifically when enabling debugging. The overridden entry point includes\n        a few extra flags to start the runtime in debug mode.\n\n        :param string runtime: Lambda function runtime name.\n        :param DebugContext debug_options: Optional. Debug context for the function (includes port, args, and path).\n        :return list: List containing the new entry points. Each element in the list is one portion of the command.\n            ie. if command is ``node index.js arg1 arg2``, then this list will be ["node", "index.js", "arg1", "arg2"]\n        '
        entry = LambdaContainer._DEFAULT_ENTRYPOINT
        if not debug_options:
            return (entry, {})
        debug_ports = debug_options.debug_ports
        container_env_vars = debug_options.container_env_vars
        if not debug_ports:
            return (entry, {})
        debug_port = debug_ports[0]
        debug_args_list = []
        if debug_options.debug_args:
            debug_args_list = debug_options.debug_args.split(' ')
        return LambdaDebugSettings.get_debug_settings(debug_port=debug_port, debug_args_list=debug_args_list, _container_env_vars=container_env_vars, runtime=runtime, options=LambdaContainer._DEBUG_ENTRYPOINT_OPTIONS)