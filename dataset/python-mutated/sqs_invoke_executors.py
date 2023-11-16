"""
Remote invoke executor implementation for SQS
"""
import json
import logging
from dataclasses import asdict, dataclass
from json.decoder import JSONDecodeError
from typing import Optional, cast
from botocore.exceptions import ClientError, ParamValidationError
from mypy_boto3_sqs import SQSClient
from samcli.lib.remote_invoke.exceptions import ErrorBotoApiCallException, InvalidResourceBotoParameterException
from samcli.lib.remote_invoke.remote_invoke_executors import BotoActionExecutor, RemoteInvokeIterableResponseType, RemoteInvokeOutputFormat, RemoteInvokeResponse
LOG = logging.getLogger(__name__)
QUEUE_URL = 'QueueUrl'
MESSAGE_BODY = 'MessageBody'
DELAY_SECONDS = 'DelaySeconds'
MESSAGE_ATTRIBUTES = 'MessageAttributes'
MESSAGE_SYSTEM_ATTRIBUTES = 'MessageSystemAttributes'

@dataclass
class SqsSendMessageTextOutput:
    """
    Dataclass that stores send_message boto3 API fields used to create
    text output.
    """
    MD5OfMessageBody: str
    MessageId: str
    MD5OfMessageAttributes: Optional[str] = None

    def get_output_response_dict(self) -> dict:
        if False:
            print('Hello World!')
        '\n        Returns a dict of existing dataclass fields.\n\n        Returns\n        -------\n        dict\n            Returns the dict of the fields that will be used as the output response for\n            text format output.\n        '
        return asdict(self, dict_factory=lambda x: {k: v for (k, v) in x if v is not None})

class SqsSendMessageExecutor(BotoActionExecutor):
    """
    Calls "send_message" method of "SQS" service with given input.
    If a file location provided, the file handle will be passed as input object.
    """
    _sqs_client: SQSClient
    _queue_url: str
    _remote_output_format: RemoteInvokeOutputFormat
    request_parameters: dict

    def __init__(self, sqs_client: SQSClient, physical_id: str, remote_output_format: RemoteInvokeOutputFormat):
        if False:
            print('Hello World!')
        self._sqs_client = sqs_client
        self._remote_output_format = remote_output_format
        self._queue_url = physical_id
        self.request_parameters = {}

    def validate_action_parameters(self, parameters: dict) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Validates the input boto parameters and prepares the parameters for calling the API.\n\n        Parameters\n        ----------\n        parameters: dict\n            Boto parameters provided as input\n        '
        try:
            for (parameter_key, parameter_value) in parameters.items():
                if parameter_key == QUEUE_URL:
                    LOG.warning('QueueUrl is defined using the value provided for resource_id argument.')
                elif parameter_key == MESSAGE_BODY:
                    LOG.warning('MessageBody is defined using the value provided for either --event or --event-file options.')
                elif parameter_key == DELAY_SECONDS:
                    self.request_parameters[parameter_key] = int(parameter_value)
                elif parameter_key in {MESSAGE_ATTRIBUTES, MESSAGE_SYSTEM_ATTRIBUTES}:
                    self.request_parameters[parameter_key] = json.loads(parameter_value)
                else:
                    self.request_parameters[parameter_key] = parameter_value
        except (ValueError, JSONDecodeError) as err:
            raise InvalidResourceBotoParameterException(f'Invalid value provided for parameter {parameter_key}', err)

    def _execute_action(self, payload: str) -> RemoteInvokeIterableResponseType:
        if False:
            i = 10
            return i + 15
        '\n        Calls "send_message" method to send a message to the SQS queue.\n\n        Parameters\n        ----------\n        payload: str\n            The MessageBody which will be sent to the SQS\n\n        Yields\n        ------\n        RemoteInvokeIterableResponseType\n            Response that is consumed by remote invoke consumers after execution\n        '
        if payload:
            self.request_parameters[MESSAGE_BODY] = payload
        else:
            self.request_parameters[MESSAGE_BODY] = '{}'
            LOG.debug('Input event not found, sending a message with MessageBody {}')
        self.request_parameters[QUEUE_URL] = self._queue_url
        LOG.debug('Calling sqs_client.send_message with QueueUrl:%s, MessageBody:%s', self.request_parameters[QUEUE_URL], self.request_parameters[MESSAGE_BODY])
        try:
            send_message_response = cast(dict, self._sqs_client.send_message(**self.request_parameters))
            if self._remote_output_format == RemoteInvokeOutputFormat.JSON:
                yield RemoteInvokeResponse(send_message_response)
            if self._remote_output_format == RemoteInvokeOutputFormat.TEXT:
                send_message_text_output = SqsSendMessageTextOutput(MD5OfMessageBody=send_message_response['MD5OfMessageBody'], MessageId=send_message_response['MessageId'], MD5OfMessageAttributes=send_message_response.get('MD5OfMessageAttributes'))
                output_data = send_message_text_output.get_output_response_dict()
                yield RemoteInvokeResponse(output_data)
        except ParamValidationError as param_val_ex:
            raise InvalidResourceBotoParameterException(f"Invalid parameter key provided. {str(param_val_ex).replace(f'{QUEUE_URL}, ', '').replace(f'{MESSAGE_BODY}, ', '')}")
        except ClientError as client_ex:
            raise ErrorBotoApiCallException(client_ex) from client_ex

def get_queue_url_from_arn(sqs_client: SQSClient, queue_name: str) -> str:
    if False:
        for i in range(10):
            print('nop')
    '\n    This function gets the queue url of the provided SQS queue name\n\n    Parameters\n    ----------\n    sqs_client: SQSClient\n        SQS client to call boto3 APIs\n    queue_name: str\n        Name of SQS queue used to get the queue_url\n\n    Returns\n    -------\n    str\n        Returns the SQS queue url\n\n    '
    try:
        output_response = sqs_client.get_queue_url(QueueName=queue_name)
        queue_url = cast(str, output_response.get(QUEUE_URL, ''))
        return queue_url
    except ClientError as client_ex:
        LOG.debug('Failed to get queue_url using the provided SQS Arn')
        raise ErrorBotoApiCallException(client_ex) from client_ex