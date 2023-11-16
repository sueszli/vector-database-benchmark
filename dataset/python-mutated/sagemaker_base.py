import json
import logging
from abc import abstractmethod, ABC
from typing import Dict, Union, List, Any
from haystack.errors import AWSConfigurationError, SageMakerConfigurationError
from haystack.lazy_imports import LazyImport
from haystack.nodes.prompt.invocation_layer.aws_base import AWSBaseInvocationLayer
from haystack.nodes.prompt.invocation_layer.handlers import DefaultPromptHandler
logger = logging.getLogger(__name__)
with LazyImport(message="Run 'pip install farm-haystack[aws]'") as boto3_import:
    import boto3
    from botocore.exceptions import ClientError

class SageMakerBaseInvocationLayer(AWSBaseInvocationLayer, ABC):
    """
    Base class for SageMaker based invocation layers.
    """

    def __init__(self, model_name_or_path: str, max_length: int=100, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(model_name_or_path, **kwargs)
        self.max_length = max_length
        model_max_length = kwargs.get('model_max_length', 1024)
        self.prompt_handler = DefaultPromptHandler(model_name_or_path='gpt2', model_max_length=model_max_length, max_length=self.max_length or 100)

    @classmethod
    @abstractmethod
    def get_test_payload(cls) -> Dict[str, Any]:
        if False:
            i = 10
            return i + 15
        '\n        Return test payload for the model.\n        '

    def _ensure_token_limit(self, prompt: Union[str, List[Dict[str, str]]]) -> Union[str, List[Dict[str, str]]]:
        if False:
            i = 10
            return i + 15
        if isinstance(prompt, List):
            raise ValueError("SageMaker invocation layer doesn't support a dictionary as prompt, only a string.")
        resize_info = self.prompt_handler(prompt)
        if resize_info['prompt_length'] != resize_info['new_prompt_length']:
            logger.warning('The prompt has been truncated from %s tokens to %s tokens so that the prompt length and answer length (%s tokens) fit within the max token limit (%s tokens). Shorten the prompt to prevent it from being cut off.', resize_info['prompt_length'], max(0, resize_info['model_max_length'] - resize_info['max_length']), resize_info['max_length'], resize_info['model_max_length'])
        return str(resize_info['resized_prompt'])

    @classmethod
    def supports(cls, model_name_or_path: str, **kwargs) -> bool:
        if False:
            i = 10
            return i + 15
        '\n        Checks whether a model_name_or_path passed down (e.g. via PromptNode) is supported by this class.\n\n        :param model_name_or_path: The model_name_or_path to check.\n        '
        if cls.aws_configured(**kwargs):
            try:
                session = cls.get_aws_session(**kwargs)
            except AWSConfigurationError as e:
                raise SageMakerConfigurationError(message=e.message) from e
            cls.check_endpoint_in_service(session, model_name_or_path)
            test_payload = cls.get_test_payload()
            supported = cls.check_model_input_format(session, model_name_or_path, test_payload, **kwargs)
            return supported
        return False

    @classmethod
    def check_endpoint_in_service(cls, session: 'boto3.Session', endpoint: str):
        if False:
            while True:
                i = 10
        '\n        Checks if the SageMaker endpoint exists and is in service.\n        :param session: The boto3 session.\n        :param endpoint: The endpoint to check.\n        '
        boto3_import.check()
        client = None
        try:
            client = session.client('sagemaker')
            response = client.describe_endpoint(EndpointName=endpoint)
            endpoint_status = response['EndpointStatus'] if 'EndpointStatus' in response else None
            if endpoint_status and endpoint_status.strip() != 'InService':
                raise SageMakerConfigurationError(f"SageMaker endpoint {endpoint} exists but is not in service. Please make sure that the endpoint is in state 'InService'.")
        except ClientError as e:
            raise SageMakerConfigurationError(f'Could not connect to {endpoint} Sagemaker endpoint. Please make sure that the endpoint exists and is accessible.') from e
        finally:
            if client:
                client.close()

    @classmethod
    def format_custom_attributes(cls, attributes: dict) -> str:
        if False:
            while True:
                i = 10
        '\n        Formats the custom attributes for the SageMaker endpoint.\n        :param attributes: The custom attributes to format.\n        :return: The formatted custom attributes.\n        '
        if attributes:
            return ';'.join((f'{k}={(str(v).lower() if isinstance(v, bool) else str(v))}' for (k, v) in attributes.items()))
        return ''

    @classmethod
    def check_model_input_format(cls, session: 'boto3.Session', endpoint: str, test_payload: Any, **kwargs):
        if False:
            print('Hello World!')
        '\n        Checks if the SageMaker endpoint supports the test_payload model input format.\n        :param session: The boto3 session.\n        :param endpoint: The endpoint to hit\n        :param test_payload: The payload to send to the endpoint\n        :return: True if the endpoint supports the test_payload model input format, False otherwise.\n        '
        boto3_import.check()
        custom_attributes = kwargs.get('aws_custom_attributes', None)
        custom_attributes = SageMakerBaseInvocationLayer.format_custom_attributes(custom_attributes)
        client = None
        try:
            client = session.client('sagemaker-runtime')
            client.invoke_endpoint(EndpointName=endpoint, Body=json.dumps(test_payload), ContentType='application/json', Accept='application/json', CustomAttributes=custom_attributes)
        except ClientError:
            return False
        finally:
            if client:
                client.close()
        return True