"""Base class for all Services that interact with Local Lambda"""
import io
import json
import logging
from typing import Tuple
from flask import Response
LOG = logging.getLogger(__name__)

class BaseLocalService:

    def __init__(self, is_debugging, port, host):
        if False:
            print('Hello World!')
        "\n        Creates a BaseLocalService class\n\n        Parameters\n        ----------\n        is_debugging bool\n            Flag to run in debug mode or not\n        port int\n            Optional. port for the service to start listening on Defaults to 3000\n        host str\n            Optional. host to start the service on Defaults to '127.0.0.1\n        "
        self.is_debugging = is_debugging
        self.port = port
        self.host = host
        self._app = None

    def create(self):
        if False:
            return 10
        '\n        Creates a Flask Application that can be started.\n        '
        raise NotImplementedError('Required method to implement')

    def run(self):
        if False:
            i = 10
            return i + 15
        '\n        This starts up the (threaded) Local Server.\n        Note: This is a **blocking call**\n\n        Raises\n        ------\n        RuntimeError\n            if the service was not created\n        '
        if not self._app:
            raise RuntimeError('The application must be created before running')
        multi_threaded = not self.is_debugging
        LOG.debug('Localhost server is starting up. Multi-threading = %s', multi_threaded)
        import flask.cli
        flask.cli.show_server_banner = lambda *args: None
        self._app.run(threaded=multi_threaded, host=self.host, port=self.port)

    @staticmethod
    def service_response(body, headers, status_code):
        if False:
            while True:
                i = 10
        '\n        Constructs a Flask Response from the body, headers, and status_code.\n\n        :param str body: Response body as a string\n        :param werkzeug.datastructures.Headers headers: headers for the response\n        :param int status_code: status_code for response\n        :return: Flask Response\n        '
        response = Response(body)
        response.headers = headers
        response.status_code = status_code
        return response

class LambdaOutputParser:

    @staticmethod
    def get_lambda_output(stdout_stream: io.StringIO) -> Tuple[str, bool]:
        if False:
            i = 10
            return i + 15
        '\n        This method will extract read the given stream and return the response from Lambda function separated out\n        from any log statements it might have outputted. Logs end up in the stdout stream if the Lambda function\n        wrote directly to stdout using System.out.println or equivalents.\n\n        Parameters\n        ----------\n        stdout_stream : io.BaseIO\n            Stream to fetch data from\n\n        Returns\n        -------\n        str\n            String data containing response from Lambda function\n        bool\n            If the response is an error/exception from the container\n        '
        lambda_response = stdout_stream.getvalue()
        is_lambda_user_error_response = LambdaOutputParser.is_lambda_error_response(lambda_response)
        return (lambda_response, is_lambda_user_error_response)

    @staticmethod
    def is_lambda_error_response(lambda_response):
        if False:
            i = 10
            return i + 15
        '\n        Check to see if the output from the container is in the form of an Error/Exception from the Lambda invoke\n\n        Parameters\n        ----------\n        lambda_response str\n            The response the container returned\n\n        Returns\n        -------\n        bool\n            True if the output matches the Error/Exception Dictionary otherwise False\n        '
        is_lambda_user_error_response = False
        lambda_response_error_dict_len = 2
        lambda_response_error_with_stacktrace_dict_len = 3
        try:
            lambda_response_dict = json.loads(lambda_response)
            if isinstance(lambda_response_dict, dict) and len(lambda_response_dict.keys() & {'errorMessage', 'errorType'}) == lambda_response_error_dict_len and (len(lambda_response_dict.keys() & {'errorMessage', 'errorType', 'stackTrace', 'cause'}) == len(lambda_response_dict) or len(lambda_response_dict) == lambda_response_error_with_stacktrace_dict_len):
                is_lambda_user_error_response = True
        except ValueError:
            pass
        return is_lambda_user_error_response