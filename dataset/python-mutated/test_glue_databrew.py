from __future__ import annotations
from typing import Generator
from unittest import mock
import pytest
from moto import mock_databrew
from airflow.providers.amazon.aws.hooks.glue_databrew import GlueDataBrewHook
from airflow.providers.amazon.aws.operators.glue_databrew import GlueDataBrewStartJobOperator
JOB_NAME = 'test_job'

@pytest.fixture
def hook() -> Generator[GlueDataBrewHook, None, None]:
    if False:
        i = 10
        return i + 15
    with mock_databrew():
        yield GlueDataBrewHook(aws_conn_id='aws_default')

class TestGlueDataBrewOperator:

    @mock.patch.object(GlueDataBrewHook, 'conn')
    @mock.patch.object(GlueDataBrewHook, 'get_waiter')
    def test_start_job_wait_for_completion(self, mock_hook_get_waiter, mock_conn):
        if False:
            for i in range(10):
                print('nop')
        TEST_RUN_ID = '12345'
        operator = GlueDataBrewStartJobOperator(task_id='task_test', job_name=JOB_NAME, wait_for_completion=True, aws_conn_id='aws_default')
        mock_conn.start_job_run(mock.MagicMock(), return_value=TEST_RUN_ID)
        operator.execute(None)
        mock_hook_get_waiter.assert_called_once_with('job_complete')

    @mock.patch.object(GlueDataBrewHook, 'conn')
    @mock.patch.object(GlueDataBrewHook, 'get_waiter')
    def test_start_job_no_wait(self, mock_hook_get_waiter, mock_conn):
        if False:
            while True:
                i = 10
        TEST_RUN_ID = '12345'
        operator = GlueDataBrewStartJobOperator(task_id='task_test', job_name=JOB_NAME, wait_for_completion=False, aws_conn_id='aws_default')
        mock_conn.start_job_run(mock.MagicMock(), return_value=TEST_RUN_ID)
        operator.execute(None)
        mock_hook_get_waiter.assert_not_called()