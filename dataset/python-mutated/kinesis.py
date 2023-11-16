"""This module contains AWS Firehose hook."""
from __future__ import annotations
from typing import Iterable
from airflow.providers.amazon.aws.hooks.base_aws import AwsBaseHook

class FirehoseHook(AwsBaseHook):
    """
    Interact with Amazon Kinesis Firehose.

    Provide thick wrapper around :external+boto3:py:class:`boto3.client("firehose") <Firehose.Client>`.

    :param delivery_stream: Name of the delivery stream

    Additional arguments (such as ``aws_conn_id``) may be specified and
    are passed down to the underlying AwsBaseHook.

    .. seealso::
        - :class:`airflow.providers.amazon.aws.hooks.base_aws.AwsBaseHook`
    """

    def __init__(self, delivery_stream: str, *args, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        self.delivery_stream = delivery_stream
        kwargs['client_type'] = 'firehose'
        super().__init__(*args, **kwargs)

    def put_records(self, records: Iterable):
        if False:
            while True:
                i = 10
        'Write batch records to Kinesis Firehose.\n\n        .. seealso::\n            - :external+boto3:py:meth:`Firehose.Client.put_record_batch`\n\n        :param records: list of records\n        '
        return self.get_conn().put_record_batch(DeliveryStreamName=self.delivery_stream, Records=records)