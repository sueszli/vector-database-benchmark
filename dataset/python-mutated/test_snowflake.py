from __future__ import annotations
import asyncio
from datetime import timedelta
from unittest import mock
import pytest
from airflow.providers.snowflake.triggers.snowflake_trigger import SnowflakeSqlApiTrigger
from airflow.triggers.base import TriggerEvent
TASK_ID = 'snowflake_check'
POLL_INTERVAL = 1.0
LIFETIME = timedelta(minutes=59)
RENEWAL_DELTA = timedelta(minutes=54)
MODULE = 'airflow.providers.snowflake'

class TestSnowflakeSqlApiTrigger:
    TRIGGER = SnowflakeSqlApiTrigger(poll_interval=POLL_INTERVAL, query_ids=['uuid'], snowflake_conn_id='test_conn', token_life_time=LIFETIME, token_renewal_delta=RENEWAL_DELTA)

    def test_snowflake_sql_trigger_serialization(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Asserts that the SnowflakeSqlApiTrigger correctly serializes its arguments\n        and classpath.\n        '
        (classpath, kwargs) = self.TRIGGER.serialize()
        assert classpath == 'airflow.providers.snowflake.triggers.snowflake_trigger.SnowflakeSqlApiTrigger'
        assert kwargs == {'poll_interval': POLL_INTERVAL, 'query_ids': ['uuid'], 'snowflake_conn_id': 'test_conn', 'token_life_time': LIFETIME, 'token_renewal_delta': RENEWAL_DELTA}

    @pytest.mark.asyncio
    @mock.patch(f'{MODULE}.triggers.snowflake_trigger.SnowflakeSqlApiTrigger.get_query_status')
    @mock.patch(f'{MODULE}.hooks.snowflake_sql_api.SnowflakeSqlApiHook.get_sql_api_query_status_async')
    async def test_snowflake_sql_trigger_running(self, mock_get_sql_api_query_status_async, mock_get_query_status):
        """Tests that the SnowflakeSqlApiTrigger in running by mocking get_query_status to true"""
        mock_get_query_status.return_value = {'status': 'running'}
        task = asyncio.create_task(self.TRIGGER.run().__anext__())
        await asyncio.sleep(0.5)
        assert task.done() is False
        asyncio.get_event_loop().stop()

    @pytest.mark.asyncio
    @mock.patch(f'{MODULE}.triggers.snowflake_trigger.SnowflakeSqlApiTrigger.get_query_status')
    @mock.patch(f'{MODULE}.hooks.snowflake_sql_api.SnowflakeSqlApiHook.get_sql_api_query_status_async')
    async def test_snowflake_sql_trigger_completed(self, mock_get_sql_api_query_status_async, mock_get_query_status):
        """
        Test SnowflakeSqlApiTrigger run method with success status and mock the get_sql_api_query_status
         result and  get_query_status to False.
        """
        mock_get_query_status.return_value = {'status': 'success', 'statement_handles': ['uuid', 'uuid1']}
        statement_query_ids = ['uuid', 'uuid1']
        mock_get_sql_api_query_status_async.return_value = {'message': 'Statement executed successfully.', 'status': 'success', 'statement_handles': statement_query_ids}
        generator = self.TRIGGER.run()
        actual = await generator.asend(None)
        assert TriggerEvent({'status': 'success', 'statement_query_ids': statement_query_ids}) == actual

    @pytest.mark.asyncio
    @mock.patch(f'{MODULE}.hooks.snowflake_sql_api.SnowflakeSqlApiHook.get_sql_api_query_status_async')
    async def test_snowflake_sql_trigger_failure_status(self, mock_get_sql_api_query_status_async):
        """Test SnowflakeSqlApiTrigger task is executed and triggered with failure status."""
        mock_response = {'status': 'error', 'message': 'An error occurred when executing the statement. Check the error code and error message for details'}
        mock_get_sql_api_query_status_async.return_value = mock_response
        generator = self.TRIGGER.run()
        actual = await generator.asend(None)
        assert TriggerEvent(mock_response) == actual

    @pytest.mark.asyncio
    @mock.patch(f'{MODULE}.hooks.snowflake_sql_api.SnowflakeSqlApiHook.get_sql_api_query_status_async')
    async def test_snowflake_sql_trigger_exception(self, mock_get_sql_api_query_status_async):
        """Tests the SnowflakeSqlApiTrigger does not fire if there is an exception."""
        mock_get_sql_api_query_status_async.side_effect = Exception('Test exception')
        task = [i async for i in self.TRIGGER.run()]
        assert len(task) == 1
        assert TriggerEvent({'status': 'error', 'message': 'Test exception'}) in task