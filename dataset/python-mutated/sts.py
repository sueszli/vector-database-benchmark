from __future__ import annotations
from airflow.providers.amazon.aws.hooks.base_aws import AwsBaseHook

class StsHook(AwsBaseHook):
    """
    Interact with AWS Security Token Service (STS).

    Provide thin wrapper around :external+boto3:py:class:`boto3.client("sts") <STS.Client>`.

    Additional arguments (such as ``aws_conn_id``) may be specified and
    are passed down to the underlying AwsBaseHook.

    .. seealso::
        - :class:`airflow.providers.amazon.aws.hooks.base_aws.AwsBaseHook`
    """

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(*args, client_type='sts', **kwargs)

    def get_account_number(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Get the account Number.\n\n        .. seealso::\n            - :external+boto3:py:meth:`STS.Client.get_caller_identity`\n        '
        try:
            return self.get_conn().get_caller_identity()['Account']
        except Exception as general_error:
            self.log.error('Failed to get the AWS Account Number, error: %s', general_error)
            raise