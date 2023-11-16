from __future__ import annotations
from unittest import mock
import pytest
from airflow.providers.apache.pig.hooks.pig import PigCliHook
from airflow.providers.apache.pig.operators.pig import PigOperator
TEST_TASK_ID = 'test_task_id'
TEST_CONTEXT_ID = 'test_context_id'
PIG = 'ls /;'

class TestPigOperator:

    def test_prepare_template(self):
        if False:
            for i in range(10):
                print('nop')
        pig = 'sh echo $DATE;'
        task_id = TEST_TASK_ID
        operator = PigOperator(pig=pig, task_id=task_id)
        operator.prepare_template()
        assert pig == operator.pig
        operator = PigOperator(pig=pig, task_id=task_id, pigparams_jinja_translate=True)
        operator.prepare_template()
        assert 'sh echo {{ DATE }};' == operator.pig

    @pytest.mark.db_test
    @mock.patch.object(PigCliHook, 'run_cli')
    def test_execute(self, mock_run_cli):
        if False:
            return 10
        pig_opts = '-x mapreduce'
        operator = PigOperator(pig=PIG, pig_opts=pig_opts, task_id=TEST_TASK_ID)
        operator.execute(context=TEST_CONTEXT_ID)
        mock_run_cli.assert_called_once_with(pig=PIG, pig_opts=pig_opts)

    @pytest.mark.db_test
    @mock.patch.object(PigCliHook, 'run_cli')
    def test_execute_default_pig_opts_to_none(self, mock_run_cli):
        if False:
            for i in range(10):
                print('nop')
        operator = PigOperator(pig=PIG, task_id=TEST_TASK_ID)
        operator.execute(context=TEST_CONTEXT_ID)
        mock_run_cli.assert_called_once_with(pig=PIG, pig_opts=None)

    @pytest.mark.db_test
    @mock.patch.object(PigCliHook, 'run_cli')
    @mock.patch.object(PigCliHook, 'kill')
    def test_on_kill(self, mock_kill, mock_rul_cli):
        if False:
            print('Hello World!')
        operator = PigOperator(pig=PIG, task_id=TEST_TASK_ID)
        operator.execute(context=TEST_CONTEXT_ID)
        operator.on_kill()
        mock_rul_cli.assert_called()
        mock_kill.assert_called()