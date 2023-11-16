from __future__ import annotations
from airflow.providers.amazon.aws.hooks.base_aws import AwsBaseHook
from airflow.utils.log.secrets_masker import mask_secret
from airflow.utils.types import NOTSET, ArgNotSet

class SsmHook(AwsBaseHook):
    """
    Interact with Amazon Systems Manager (SSM).

    Provide thin wrapper around :external+boto3:py:class:`boto3.client("ssm") <SSM.Client>`.

    Additional arguments (such as ``aws_conn_id``) may be specified and
    are passed down to the underlying AwsBaseHook.

    .. seealso::
        - :class:`airflow.providers.amazon.aws.hooks.base_aws.AwsBaseHook`
    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        kwargs['client_type'] = 'ssm'
        super().__init__(*args, **kwargs)

    def get_parameter_value(self, parameter: str, default: str | ArgNotSet=NOTSET) -> str:
        if False:
            print('Hello World!')
        '\n        Return the provided Parameter or an optional default; if it is encrypted, then decrypt and mask.\n\n        .. seealso::\n            - :external+boto3:py:meth:`SSM.Client.get_parameter`\n\n        :param parameter: The SSM Parameter name to return the value for.\n        :param default: Optional default value to return if none is found.\n        '
        try:
            param = self.conn.get_parameter(Name=parameter, WithDecryption=True)['Parameter']
            value = param['Value']
            if param['Type'] == 'SecureString':
                mask_secret(value)
            return value
        except self.conn.exceptions.ParameterNotFound:
            if isinstance(default, ArgNotSet):
                raise
            return default