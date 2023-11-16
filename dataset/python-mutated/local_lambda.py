"""
Implementation of Local Lambda runner
"""
import logging
import os
from typing import Any, Dict, Optional, cast
import boto3
from botocore.credentials import Credentials
from samcli.commands.local.lib.debug_context import DebugContext
from samcli.commands.local.lib.exceptions import InvalidIntermediateImageError, NoPrivilegeException, OverridesNotWellDefinedError, UnsupportedInlineCodeError
from samcli.lib.providers.provider import Function
from samcli.lib.providers.sam_function_provider import SamFunctionProvider
from samcli.lib.utils.architecture import validate_architecture_runtime
from samcli.lib.utils.codeuri import resolve_code_path
from samcli.lib.utils.packagetype import IMAGE, ZIP
from samcli.lib.utils.stream_writer import StreamWriter
from samcli.local.docker.container import ContainerConnectionTimeoutException, ContainerResponseException
from samcli.local.lambdafn.config import FunctionConfig
from samcli.local.lambdafn.env_vars import EnvironmentVariables
from samcli.local.lambdafn.exceptions import FunctionNotFound
from samcli.local.lambdafn.runtime import LambdaRuntime
LOG = logging.getLogger(__name__)

class LocalLambdaRunner:
    """
    Runs Lambda functions locally. This class is a wrapper around the `samcli.local` library which takes care
    of actually running the function on a Docker container.
    """
    MAX_DEBUG_TIMEOUT = 36000
    WIN_ERROR_CODE = 1314

    def __init__(self, local_runtime: LambdaRuntime, function_provider: SamFunctionProvider, cwd: str, aws_profile: Optional[str]=None, aws_region: Optional[str]=None, env_vars_values: Optional[Dict[Any, Any]]=None, debug_context: Optional[DebugContext]=None, container_host: Optional[str]=None, container_host_interface: Optional[str]=None) -> None:
        if False:
            while True:
                i = 10
        '\n        Initializes the class\n\n        :param samcli.local.lambdafn.runtime.LambdaRuntime local_runtime: Lambda runtime capable of running a function\n        :param samcli.commands.local.lib.provider.FunctionProvider function_provider: Provider that can return a\n            Lambda function\n        :param string cwd: Current working directory. We will resolve all function CodeURIs relative to this directory.\n        :param string aws_profile: Optional. Name of the profile to fetch AWS credentials from.\n        :param string aws_region: Optional. AWS Region to use.\n        :param dict env_vars_values: Optional. Dictionary containing values of environment variables.\n        :param DebugContext debug_context: Optional. Debug context for the function (includes port, args, and path).\n        :param string container_host: Optional. Host of locally emulated Lambda container\n        :param string container_host_interface: Optional. Interface that Docker host binds ports to\n        '
        self.local_runtime = local_runtime
        self.provider = function_provider
        self.cwd = cwd
        self.aws_profile = aws_profile
        self.aws_region = aws_region
        self.env_vars_values = env_vars_values or {}
        self.debug_context = debug_context
        self._boto3_session_creds: Optional[Credentials] = None
        self._boto3_region: Optional[str] = None
        self.container_host = container_host
        self.container_host_interface = container_host_interface

    def invoke(self, function_identifier: str, event: str, stdout: Optional[StreamWriter]=None, stderr: Optional[StreamWriter]=None) -> None:
        if False:
            return 10
        '\n        Find the Lambda function with given name and invoke it. Pass the given event to the function and return\n        response through the given streams.\n\n        This function will block until either the function completes or times out.\n\n        Parameters\n        ----------\n        function_identifier str\n            Identifier of the Lambda function to invoke, it can be logicalID, function name or full path\n        event str\n            Event data passed to the function. Must be a valid JSON String.\n        stdout samcli.lib.utils.stream_writer.StreamWriter\n            Stream writer to write the output of the Lambda function to.\n        stderr samcli.lib.utils.stream_writer.StreamWriter\n            Stream writer to write the Lambda runtime logs to.\n\n        Raises\n        ------\n        FunctionNotfound\n            When we cannot find a function with the given name\n        '
        function = self.provider.get(function_identifier)
        if not function:
            all_function_full_paths = [f.full_path for f in self.provider.get_all()]
            available_function_message = '{} not found. Possible options in your template: {}'.format(function_identifier, all_function_full_paths)
            LOG.info(available_function_message)
            raise FunctionNotFound("Unable to find a Function with name '{}'".format(function_identifier))
        LOG.debug("Found one Lambda function with name '%s'", function_identifier)
        if function.packagetype == ZIP:
            if function.inlinecode:
                raise UnsupportedInlineCodeError(f'Inline code is not supported for sam local commands. Please write your code in a separate file for the function {function.function_id}.')
            LOG.info('Invoking %s (%s)', function.handler, function.runtime)
        elif function.packagetype == IMAGE:
            if not function.imageuri:
                raise InvalidIntermediateImageError(f'ImageUri not provided for Function: {function_identifier} of PackageType: {function.packagetype}')
            LOG.info('Invoking Container created from %s', function.imageuri)
        validate_architecture_runtime(function)
        config = self.get_invoke_config(function)
        try:
            self.local_runtime.invoke(config, event, debug_context=self.debug_context, stdout=stdout, stderr=stderr, container_host=self.container_host, container_host_interface=self.container_host_interface)
        except ContainerResponseException:
            LOG.info('No response from invoke container for %s', function.name)
        except ContainerConnectionTimeoutException as e:
            LOG.info(str(e))
        except OSError as os_error:
            if getattr(os_error, 'winerror', None) == self.WIN_ERROR_CODE:
                raise NoPrivilegeException('Administrator, Windows Developer Mode, or SeCreateSymbolicLinkPrivilege is required to create symbolic link for files: {}, {}'.format(os_error.filename, os_error.filename2)) from os_error
            raise

    def is_debugging(self) -> bool:
        if False:
            return 10
        '\n        Are we debugging the invoke?\n\n        Returns\n        -------\n        bool\n            True, if we are debugging the invoke ie. the Docker container will break into the debugger and wait for\n            attach\n        '
        return bool(self.debug_context)

    def get_invoke_config(self, function: Function) -> FunctionConfig:
        if False:
            print('Hello World!')
        '\n        Returns invoke configuration to pass to Lambda Runtime to invoke the given function\n\n        :param samcli.commands.local.lib.provider.Function function: Lambda function to generate the configuration for\n        :return samcli.local.lambdafn.config.FunctionConfig: Function configuration to pass to Lambda runtime\n        '
        env_vars = self._make_env_vars(function)
        code_abs_path = None
        if function.packagetype == ZIP:
            code_abs_path = resolve_code_path(self.cwd, function.codeuri)
            LOG.debug('Resolved absolute path to code is %s', code_abs_path)
        function_timeout = function.timeout
        if self.is_debugging():
            function_timeout = self.MAX_DEBUG_TIMEOUT
        return FunctionConfig(name=function.name, full_path=function.full_path, runtime=function.runtime, handler=function.handler, imageuri=function.imageuri, imageconfig=function.imageconfig, packagetype=function.packagetype, code_abs_path=code_abs_path, layers=function.layers, architecture=function.architecture, memory=function.memory, timeout=function_timeout, env_vars=env_vars, runtime_management_config=function.runtime_management_config)

    def _make_env_vars(self, function: Function) -> EnvironmentVariables:
        if False:
            while True:
                i = 10
        'Returns the environment variables configuration for this function\n\n        Priority order for environment variables (high to low):\n        1. Function specific env vars from json file\n        2. Global env vars from json file\n\n        Parameters\n        ----------\n        function : samcli.commands.local.lib.provider.Function\n            Lambda function to generate the configuration for\n\n        Returns\n        -------\n        samcli.local.lambdafn.env_vars.EnvironmentVariables\n            Environment variable configuration for this function\n\n        Raises\n        ------\n        samcli.commands.local.lib.exceptions.OverridesNotWellDefinedError\n            If the environment dict is in the wrong format to process environment vars\n\n        '
        function_id = function.function_id
        logical_id = function.name
        function_name = function.functionname
        full_path = function.full_path
        variables = None
        if isinstance(function.environment, dict) and 'Variables' in function.environment:
            variables = function.environment['Variables']
        else:
            LOG.debug("No environment variables found for function '%s'", logical_id)
        for env_var_value in self.env_vars_values.values():
            if not isinstance(env_var_value, dict):
                reason = 'Environment variables {} in incorrect format'.format(env_var_value)
                LOG.debug(reason)
                raise OverridesNotWellDefinedError(reason)
        overrides = {}
        if 'Parameters' in self.env_vars_values:
            LOG.debug('Environment variables data found in the CloudFormation parameter file format')
            parameter_result = self.env_vars_values.get('Parameters', {})
            overrides.update(parameter_result)
        fn_file_env_vars = self.env_vars_values.get(logical_id, None) or self.env_vars_values.get(function_id, None) or self.env_vars_values.get(function_name, None) or self.env_vars_values.get(full_path, None)
        if fn_file_env_vars:
            LOG.debug('Environment variables data found for specific function in standard format')
            overrides.update(fn_file_env_vars)
        shell_env = os.environ
        aws_creds = self.get_aws_creds()
        return EnvironmentVariables(function.name, function.memory, function.timeout, function.handler, variables=variables, shell_env_values=shell_env, override_values=overrides, aws_creds=aws_creds)

    def _get_session_creds(self) -> Optional[Credentials]:
        if False:
            i = 10
            return i + 15
        if self._boto3_session_creds is None:
            LOG.debug("Loading AWS credentials from session with profile '%s'", self.aws_profile)
            session = boto3.session.Session(profile_name=cast(str, self.aws_profile), region_name=cast(str, self.aws_region))
            if hasattr(session, 'region_name') and session.region_name:
                self._boto3_region = session.region_name
            if session:
                self._boto3_session_creds = session.get_credentials()
        return self._boto3_session_creds

    def get_aws_creds(self) -> Dict[str, str]:
        if False:
            print('Hello World!')
        '\n        Returns AWS credentials obtained from the shell environment or given profile\n\n        :return dict: A dictionary containing credentials. This dict has the structure\n             {"region": "", "key": "", "secret": "", "sessiontoken": ""}. If credentials could not be resolved,\n             this returns None\n        '
        result: Dict[str, str] = {}
        creds = self._get_session_creds()
        if self._boto3_region:
            result['region'] = self._boto3_region
        if not creds:
            return result
        if hasattr(creds, 'access_key') and creds.access_key:
            result['key'] = creds.access_key
        if hasattr(creds, 'secret_key') and creds.secret_key:
            result['secret'] = creds.secret_key
        if hasattr(creds, 'token') and creds.token:
            result['sessiontoken'] = creds.token
        return result