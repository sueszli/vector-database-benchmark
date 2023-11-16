"""Local Lambda Service that only invokes a function"""
import io
import json
import logging
from flask import Flask, request
from werkzeug.routing import BaseConverter
from samcli.commands.local.lib.exceptions import UnsupportedInlineCodeError
from samcli.lib.utils.stream_writer import StreamWriter
from samcli.local.lambdafn.exceptions import FunctionNotFound
from samcli.local.services.base_local_service import BaseLocalService, LambdaOutputParser
from .lambda_error_responses import LambdaErrorResponses
LOG = logging.getLogger(__name__)

class FunctionNamePathConverter(BaseConverter):
    regex = '.+'
    weight = 300
    part_isolating = False

    def to_python(self, value):
        if False:
            i = 10
            return i + 15
        return value

    def to_url(self, value):
        if False:
            i = 10
            return i + 15
        return value

class LocalLambdaInvokeService(BaseLocalService):

    def __init__(self, lambda_runner, port, host, stderr=None):
        if False:
            while True:
                i = 10
        '\n        Creates a Local Lambda Service that will only response to invoking a function\n\n        Parameters\n        ----------\n        lambda_runner samcli.commands.local.lib.local_lambda.LocalLambdaRunner\n            The Lambda runner class capable of invoking the function\n        port int\n            Optional. port for the service to start listening on\n        host str\n            Optional. host to start the service on\n        stderr io.BaseIO\n            Optional stream where the stderr from Docker container should be written to\n        '
        super().__init__(lambda_runner.is_debugging(), port=port, host=host)
        self.lambda_runner = lambda_runner
        self.stderr = stderr

    def create(self):
        if False:
            i = 10
            return i + 15
        '\n        Creates a Flask Application that can be started.\n        '
        self._app = Flask(__name__)
        self._app.url_map.converters['function_path'] = FunctionNamePathConverter
        path = '/2015-03-31/functions/<function_path:function_name>/invocations'
        self._app.add_url_rule(path, endpoint=path, view_func=self._invoke_request_handler, methods=['POST'], provide_automatic_options=False)
        self._app.before_request(LocalLambdaInvokeService.validate_request)
        self._construct_error_handling()

    @staticmethod
    def validate_request():
        if False:
            for i in range(10):
                print('nop')
        "\n        Validates the incoming request\n\n        The following are invalid\n            1. The Request data is not json serializable\n            2. Query Parameters are sent to the endpoint\n            3. The Request Content-Type is not application/json\n            4. 'X-Amz-Log-Type' header is not 'None'\n            5. 'X-Amz-Invocation-Type' header is not 'RequestResponse'\n\n        Returns\n        -------\n        flask.Response\n            If the request is not valid a flask Response is returned\n\n        None:\n            If the request passes all validation\n        "
        flask_request = request
        request_data = flask_request.get_data()
        if not request_data:
            request_data = b'{}'
        request_data = request_data.decode('utf-8')
        try:
            json.loads(request_data)
        except ValueError as json_error:
            LOG.debug('Request body was not json. Exception: %s', str(json_error))
            return LambdaErrorResponses.invalid_request_content('Could not parse request body into json: No JSON object could be decoded')
        if flask_request.args:
            LOG.debug('Query parameters are in the request but not supported')
            return LambdaErrorResponses.invalid_request_content('Query Parameters are not supported')
        request_headers = flask_request.headers
        log_type = request_headers.get('X-Amz-Log-Type', 'None')
        if log_type != 'None':
            LOG.debug('log-type: %s is not supported. None is only supported.', log_type)
            return LambdaErrorResponses.not_implemented_locally('log-type: {} is not supported. None is only supported.'.format(log_type))
        invocation_type = request_headers.get('X-Amz-Invocation-Type', 'RequestResponse')
        if invocation_type != 'RequestResponse':
            LOG.warning('invocation-type: %s is not supported. RequestResponse is only supported.', invocation_type)
            return LambdaErrorResponses.not_implemented_locally('invocation-type: {} is not supported. RequestResponse is only supported.'.format(invocation_type))
        return None

    def _construct_error_handling(self):
        if False:
            print('Hello World!')
        '\n        Updates the Flask app with Error Handlers for different Error Codes\n\n        '
        self._app.register_error_handler(500, LambdaErrorResponses.generic_service_exception)
        self._app.register_error_handler(404, LambdaErrorResponses.generic_path_not_found)
        self._app.register_error_handler(405, LambdaErrorResponses.generic_method_not_allowed)

    def _invoke_request_handler(self, function_name):
        if False:
            return 10
        '\n        Request Handler for the Local Lambda Invoke path. This method is responsible for understanding the incoming\n        request and invoking the Local Lambda Function\n\n        Parameters\n        ----------\n        function_name str\n            Name of the function to invoke\n\n        Returns\n        -------\n        A Flask Response response object as if it was returned from Lambda\n        '
        flask_request = request
        request_data = flask_request.get_data()
        if not request_data:
            request_data = b'{}'
        request_data = request_data.decode('utf-8')
        stdout_stream = io.StringIO()
        stdout_stream_writer = StreamWriter(stdout_stream, auto_flush=True)
        try:
            self.lambda_runner.invoke(function_name, request_data, stdout=stdout_stream_writer, stderr=self.stderr)
        except FunctionNotFound:
            LOG.debug('%s was not found to invoke.', function_name)
            return LambdaErrorResponses.resource_not_found(function_name)
        except UnsupportedInlineCodeError:
            return LambdaErrorResponses.not_implemented_locally('Inline code is not supported for sam local commands. Please write your code in a separate file.')
        (lambda_response, is_lambda_user_error_response) = LambdaOutputParser.get_lambda_output(stdout_stream)
        if is_lambda_user_error_response:
            return self.service_response(lambda_response, {'Content-Type': 'application/json', 'x-amz-function-error': 'Unhandled'}, 200)
        return self.service_response(lambda_response, {'Content-Type': 'application/json'}, 200)