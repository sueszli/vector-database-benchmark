from __future__ import annotations
import copy
from google.api_core.gapic_v1.method import DEFAULT, _MethodDefault
from google.api_core.retry import Retry
from airflow.providers.google.cloud.operators.cloud_base import GoogleCloudBaseOperator
TASK_ID = 'task-id'

class GoogleSampleOperator(GoogleCloudBaseOperator):

    def __init__(self, retry: Retry | _MethodDefault=DEFAULT, config: dict | None=None, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(**kwargs)
        self.retry = retry
        self.config = config

class TestGoogleCloudBaseOperator:

    def test_handles_deepcopy_with_method_default(self):
        if False:
            return 10
        op = GoogleSampleOperator(task_id=TASK_ID)
        copied_op = copy.deepcopy(op)
        assert copied_op.retry == DEFAULT
        assert copied_op.config is None

    def test_handles_deepcopy_with_non_default_retry(self):
        if False:
            return 10
        op = GoogleSampleOperator(task_id=TASK_ID, retry=Retry(deadline=30), config={'config': 'value'})
        copied_op = copy.deepcopy(op)
        assert copied_op.retry.deadline == 30
        assert copied_op.config == {'config': 'value'}