import logging
from abc import ABC
from typing import Optional
from haystack.errors import AWSConfigurationError
from haystack.lazy_imports import LazyImport
from haystack.nodes.prompt.invocation_layer import PromptModelInvocationLayer
logger = logging.getLogger(__name__)
with LazyImport(message="Run 'pip install farm-haystack[aws]'") as boto3_import:
    import boto3
    from botocore.exceptions import BotoCoreError
AWS_CONFIGURATION_KEYS = ['aws_access_key_id', 'aws_secret_access_key', 'aws_session_token', 'aws_region_name', 'aws_profile_name']

class AWSBaseInvocationLayer(PromptModelInvocationLayer, ABC):
    """
    Base class for AWS based invocation layers.
    """

    @classmethod
    def aws_configured(cls, **kwargs) -> bool:
        if False:
            i = 10
            return i + 15
        '\n        Checks whether this invocation layer is active.\n        :param kwargs: The kwargs passed down to the invocation layer.\n        :return: True if the invocation layer is active, False otherwise.\n        '
        aws_config_provided = any((key in kwargs for key in AWS_CONFIGURATION_KEYS))
        return aws_config_provided

    @classmethod
    def get_aws_session(cls, aws_access_key_id: Optional[str]=None, aws_secret_access_key: Optional[str]=None, aws_session_token: Optional[str]=None, aws_region_name: Optional[str]=None, aws_profile_name: Optional[str]=None, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Creates an AWS Session with the given parameters.\n        Checks if the provided AWS credentials are valid and can be used to connect to AWS.\n\n        :param aws_access_key_id: AWS access key ID.\n        :param aws_secret_access_key: AWS secret access key.\n        :param aws_session_token: AWS session token.\n        :param aws_region_name: AWS region name.\n        :param aws_profile_name: AWS profile name.\n        :param kwargs: The kwargs passed down to the service client. Supported kwargs depend on the model chosen.\n            See https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters.html.\n        :raises AWSConfigurationError: If the provided AWS credentials are invalid.\n        :return: The created AWS session.\n        '
        boto3_import.check()
        try:
            return boto3.Session(aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key, aws_session_token=aws_session_token, region_name=aws_region_name, profile_name=aws_profile_name)
        except BotoCoreError as e:
            provided_aws_config = {k: v for (k, v) in kwargs.items() if k in AWS_CONFIGURATION_KEYS}
            raise AWSConfigurationError(f'Failed to initialize the session with provided AWS credentials {provided_aws_config}') from e