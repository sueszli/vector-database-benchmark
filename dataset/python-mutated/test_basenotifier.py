from __future__ import annotations
from typing import TYPE_CHECKING
from unittest.mock import MagicMock
import jinja2
import pytest
from airflow.notifications.basenotifier import BaseNotifier
from airflow.operators.empty import EmptyOperator
pytestmark = pytest.mark.db_test
if TYPE_CHECKING:
    from airflow.utils.context import Context

class MockNotifier(BaseNotifier):
    """MockNotifier class for testing"""
    template_fields = ('message',)
    template_ext = ('.txt',)

    def __init__(self, message: str | None='This is a test message'):
        if False:
            while True:
                i = 10
        super().__init__()
        self.message = message

    def notify(self, context: Context) -> None:
        if False:
            return 10
        pass

class TestBaseNotifier:

    def test_render_message_with_message(self, dag_maker):
        if False:
            while True:
                i = 10
        with dag_maker('test_render_message_with_message') as dag:
            EmptyOperator(task_id='test_id')
        notifier = MockNotifier(message='Hello {{ dag.dag_id }}')
        context: Context = {'dag': dag}
        notifier.render_template_fields(context)
        assert notifier.message == 'Hello test_render_message_with_message'

    def test_render_message_with_template(self, dag_maker, caplog):
        if False:
            i = 10
            return i + 15
        with dag_maker('test_render_message_with_template') as dag:
            EmptyOperator(task_id='test_id')
        notifier = MockNotifier(message='test.txt')
        context: Context = {'dag': dag}
        with pytest.raises(jinja2.exceptions.TemplateNotFound):
            notifier.render_template_fields(context)

    def test_render_message_with_template_works(self, dag_maker, caplog):
        if False:
            print('Hello World!')
        with dag_maker('test_render_message_with_template_works') as dag:
            EmptyOperator(task_id='test_id')
        notifier = MockNotifier(message='test_notifier.txt')
        context: Context = {'dag': dag}
        notifier.render_template_fields(context)
        assert notifier.message == 'Hello test_render_message_with_template_works'

    def test_notifier_call_with_passed_context(self, dag_maker, caplog):
        if False:
            for i in range(10):
                print('nop')
        with dag_maker('test_render_message_with_template_works') as dag:
            EmptyOperator(task_id='test_id')
        notifier = MockNotifier(message='Hello {{ dag.dag_id }}')
        notifier.notify = MagicMock()
        context: Context = {'dag': dag}
        notifier(context)
        notifier.notify.assert_called_once_with({'dag': dag, 'message': 'Hello {{ dag.dag_id }}'})
        assert notifier.message == 'Hello test_render_message_with_template_works'

    def test_notifier_call_with_prepared_context(self, dag_maker, caplog):
        if False:
            print('Hello World!')
        with dag_maker('test_render_message_with_template_works'):
            EmptyOperator(task_id='test_id')
        notifier = MockNotifier(message='task: {{ task_list[0] }}')
        notifier.notify = MagicMock()
        notifier(None, ['some_task'], None, None, None)
        notifier.notify.assert_called_once_with({'dag': None, 'task_list': ['some_task'], 'blocking_task_list': None, 'slas': None, 'blocking_tis': None, 'message': 'task: {{ task_list[0] }}'})
        assert notifier.message == 'task: some_task'