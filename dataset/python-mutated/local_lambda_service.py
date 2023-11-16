"""
Connects the CLI with Local Lambda Invoke Service.
"""
import logging
from samcli.local.lambda_service.local_lambda_invoke_service import LocalLambdaInvokeService
LOG = logging.getLogger(__name__)

class LocalLambdaService:
    """
    Implementation of Local Lambda Invoke Service that is capable of serving the invoke path to your Lambda Functions
    that are defined in a SAM file.
    """

    def __init__(self, lambda_invoke_context, port, host):
        if False:
            i = 10
            return i + 15
        '\n        Initialize the Local Lambda Invoke service.\n\n        :param samcli.commands.local.cli_common.invoke_context.InvokeContext lambda_invoke_context: Context object\n            that can help with Lambda invocation\n        :param int port: Port to listen on\n        :param string host: Local hostname or IP address to bind to\n        '
        self.port = port
        self.host = host
        self.lambda_runner = lambda_invoke_context.local_lambda_runner
        self.stderr_stream = lambda_invoke_context.stderr

    def start(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Creates and starts the Local Lambda Invoke service. This method will block until the service is stopped\n        manually using an interrupt. After the service is started, callers can make HTTP requests to the endpoint\n        to invoke the Lambda function and receive a response.\n\n        NOTE: This is a blocking call that will not return until the thread is interrupted with SIGINT/SIGTERM\n        '
        service = LocalLambdaInvokeService(lambda_runner=self.lambda_runner, port=self.port, host=self.host, stderr=self.stderr_stream)
        service.create()
        LOG.info('Starting the Local Lambda Service. You can now invoke your Lambda Functions defined in your template through the endpoint.')
        service.run()