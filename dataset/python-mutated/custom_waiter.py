"""
Base class for implementing custom waiters for services that don't already have
prebuilt waiters. This class leverages botocore waiter code.
"""
from enum import Enum
import logging
import botocore.waiter
logger = logging.getLogger(__name__)

class WaitState(Enum):
    SUCCESS = 'success'
    FAILURE = 'failure'

class CustomWaiter:
    """
    Base class for a custom waiter that leverages botocore's waiter code. Waiters
    poll an operation, with a specified delay between each polling attempt, until
    either an accepted result is returned or the number of maximum attempts is reached.

    To use, implement a subclass that passes the specific operation, arguments,
    and acceptors to the superclass.

    For example, to implement a custom waiter for the transcription client that
    waits for both success and failure outcomes of the get_transcription_job function,
    create a class like the following:

        class TranscribeCompleteWaiter(CustomWaiter):
        def __init__(self, client):
            super().__init__(
                'TranscribeComplete', 'GetTranscriptionJob',
                'TranscriptionJob.TranscriptionJobStatus',
                {'COMPLETED': WaitState.SUCCESS, 'FAILED': WaitState.FAILURE},
                client)

        def wait(self, job_name):
            self._wait(TranscriptionJobName=job_name)

    """

    def __init__(self, name, operation, argument, acceptors, client, delay=10, max_tries=60, matcher='path'):
        if False:
            print('Hello World!')
        "\n        Subclasses should pass specific operations, arguments, and acceptors to\n        their superclass.\n\n        :param name: The name of the waiter. This can be any descriptive string.\n        :param operation: The operation to wait for. This must match the casing of\n                          the underlying operation model, which is typically in\n                          CamelCase.\n        :param argument: The dict keys used to access the result of the operation, in\n                         dot notation. For example, 'Job.Status' will access\n                         result['Job']['Status'].\n        :param acceptors: The list of acceptors that indicate the wait is over. These\n                          can indicate either success or failure. The acceptor values\n                          are compared to the result of the operation after the\n                          argument keys are applied.\n        :param client: The Boto3 client.\n        :param delay: The number of seconds to wait between each call to the operation.\n        :param max_tries: The maximum number of tries before exiting.\n        :param matcher: The kind of matcher to use.\n        "
        self.name = name
        self.operation = operation
        self.argument = argument
        self.client = client
        self.waiter_model = botocore.waiter.WaiterModel({'version': 2, 'waiters': {name: {'delay': delay, 'operation': operation, 'maxAttempts': max_tries, 'acceptors': [{'state': state.value, 'matcher': matcher, 'argument': argument, 'expected': expected} for (expected, state) in acceptors.items()]}}})
        self.waiter = botocore.waiter.create_waiter_with_client(self.name, self.waiter_model, self.client)

    def __call__(self, parsed, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Handles the after-call event by logging information about the operation and its\n        result.\n\n        :param parsed: The parsed response from polling the operation.\n        :param kwargs: Not used, but expected by the caller.\n        '
        status = parsed
        for key in self.argument.split('.'):
            if key.endswith('[]'):
                status = status.get(key[:-2])[0]
            else:
                status = status.get(key)
        logger.info('Waiter %s called %s, got %s.', self.name, self.operation, status)

    def _wait(self, **kwargs):
        if False:
            print('Hello World!')
        '\n        Registers for the after-call event and starts the botocore wait loop.\n\n        :param kwargs: Keyword arguments that are passed to the operation being polled.\n        '
        event_name = f'after-call.{self.client.meta.service_model.service_name}'
        self.client.meta.events.register(event_name, self)
        self.waiter.wait(**kwargs)
        self.client.meta.events.unregister(event_name, self)