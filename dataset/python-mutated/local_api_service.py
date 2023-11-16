"""
Connects the CLI with Local API Gateway service.
"""
import logging
import os
from samcli.commands.local.lib.exceptions import NoApisDefined
from samcli.lib.providers.api_provider import ApiProvider
from samcli.local.apigw.local_apigw_service import LocalApigwService
LOG = logging.getLogger(__name__)

class LocalApiService:
    """
    Implementation of Local API service that is capable of serving API defined in a configuration file that invoke a
    Lambda function.
    """

    def __init__(self, lambda_invoke_context, port, host, static_dir):
        if False:
            return 10
        '\n        Initialize the local API service.\n\n        :param samcli.commands.local.cli_common.invoke_context.InvokeContext lambda_invoke_context: Context object\n            that can help with Lambda invocation\n        :param int port: Port to listen on\n        :param string host: Local hostname or IP address to bind to\n        :param string static_dir: Optional, directory from which static files will be mounted\n        '
        self.port = port
        self.host = host
        self.static_dir = static_dir
        self.cwd = lambda_invoke_context.get_cwd()
        self.api_provider = ApiProvider(lambda_invoke_context.stacks, cwd=self.cwd)
        self.lambda_runner = lambda_invoke_context.local_lambda_runner
        self.stderr_stream = lambda_invoke_context.stderr

    def start(self):
        if False:
            print('Hello World!')
        '\n        Creates and starts the local API Gateway service. This method will block until the service is stopped\n        manually using an interrupt. After the service is started, callers can make HTTP requests to the endpoint\n        to invoke the Lambda function and receive a response.\n\n        NOTE: This is a blocking call that will not return until the thread is interrupted with SIGINT/SIGTERM\n        '
        if not self.api_provider.api.routes:
            raise NoApisDefined('No APIs available in template')
        static_dir_path = self._make_static_dir_path(self.cwd, self.static_dir)
        service = LocalApigwService(api=self.api_provider.api, lambda_runner=self.lambda_runner, static_dir=static_dir_path, port=self.port, host=self.host, stderr=self.stderr_stream)
        service.create()
        self._print_routes(self.api_provider.api.routes, self.host, self.port)
        LOG.info('You can now browse to the above endpoints to invoke your functions. You do not need to restart/reload SAM CLI while working on your functions, changes will be reflected instantly/automatically. If you used sam build before running local commands, you will need to re-run sam build for the changes to be picked up. You only need to restart SAM CLI if you update your AWS SAM template')
        service.run()

    @staticmethod
    def _print_routes(routes, host, port):
        if False:
            i = 10
            return i + 15
        '\n        Helper method to print the APIs that will be mounted. This method is purely for printing purposes.\n        This method takes in a list of Route Configurations and prints out the Routes grouped by path.\n        Grouping routes by Function Name + Path is the bulk of the logic.\n\n        Example output:\n            Mounting Product at http://127.0.0.1:3000/path1/bar [GET, POST, DELETE]\n            Mounting Product at http://127.0.0.1:3000/path2/bar [HEAD]\n\n        :param list(Route) routes:\n            List of routes grouped by the same function_name and path\n        :param string host:\n            Host name where the service is running\n        :param int port:\n            Port number where the service is running\n        :returns list(string):\n            List of lines that were printed to the console. Helps with testing\n        '
        print_lines = []
        for route in routes:
            methods_str = '[{}]'.format(', '.join(route.methods))
            output = 'Mounting {} at http://{}:{}{} {}'.format(route.function_name, host, port, route.path, methods_str)
            print_lines.append(output)
            LOG.info(output)
        return print_lines

    @staticmethod
    def _make_static_dir_path(cwd, static_dir):
        if False:
            print('Hello World!')
        '\n        This method returns the path to the directory where static files are to be served from. If static_dir is a\n        relative path, then it is resolved to be relative to the current working directory. If no static directory is\n        provided, or if the resolved directory does not exist, this method will return None\n\n        :param string cwd: Current working directory relative to which we will resolve the static directory\n        :param string static_dir: Path to the static directory\n        :return string: Path to the static directory, if it exists. None, otherwise\n        '
        if not static_dir:
            return None
        static_dir_path = os.path.join(cwd, static_dir)
        if os.path.exists(static_dir_path):
            LOG.info('Mounting static files from %s at /', static_dir_path)
            return static_dir_path
        return None