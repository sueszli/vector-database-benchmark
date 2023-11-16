from __future__ import annotations
from typing import Sequence
from airflow.providers.amazon.aws.utils.mixins import AwsBaseHookMixin, AwsHookParams, AwsHookType, aws_template_fields
from airflow.sensors.base import BaseSensorOperator

class AwsBaseSensor(BaseSensorOperator, AwsBaseHookMixin[AwsHookType]):
    """Base AWS (Amazon) Sensor Class for build sensors in top of AWS Hooks.

    .. warning::
        Only for internal usage, this class might be changed, renamed or removed in the future
        without any further notice.

    Examples:
     .. code-block:: python

        from airflow.providers.amazon.aws.hooks.foo_bar import FooBarThinHook, FooBarThickHook


        class AwsFooBarSensor(AwsBaseSensor[FooBarThinHook]):
            aws_hook_class = FooBarThinHook

            def poke(self, context):
                pass


        class AwsFooBarSensor(AwsBaseSensor[FooBarThickHook]):
            aws_hook_class = FooBarThickHook

            def __init__(self, *, spam: str, **kwargs):
                super().__init__(**kwargs)
                self.spam = spam

            @property
            def _hook_parameters(self):
                return {**super()._hook_parameters, "spam": self.spam}

            def poke(self, context):
                pass

    :param aws_conn_id: The Airflow connection used for AWS credentials.
        If this is None or empty then the default boto3 behaviour is used. If
        running Airflow in a distributed manner and aws_conn_id is None or
        empty, then default boto3 configuration would be used (and must be
        maintained on each worker node).
    :param region_name: AWS region_name. If not specified then the default boto3 behaviour is used.
    :param verify: Whether or not to verify SSL certificates. See:
        https://boto3.amazonaws.com/v1/documentation/api/latest/reference/core/session.html
    :param botocore_config: Configuration dictionary (key-values) for botocore client. See:
        https://botocore.amazonaws.com/v1/documentation/api/latest/reference/config.html
    :meta private:
    """
    template_fields: Sequence[str] = aws_template_fields()

    def __init__(self, *, aws_conn_id: str | None='aws_default', region_name: str | None=None, verify: bool | str | None=None, botocore_config: dict | None=None, **kwargs):
        if False:
            print('Hello World!')
        hook_params = AwsHookParams.from_constructor(aws_conn_id, region_name, verify, botocore_config, additional_params=kwargs)
        super().__init__(**kwargs)
        self.aws_conn_id = hook_params.aws_conn_id
        self.region_name = hook_params.region_name
        self.verify = hook_params.verify
        self.botocore_config = hook_params.botocore_config
        self.validate_attributes()