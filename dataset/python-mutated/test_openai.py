from __future__ import annotations
from unittest.mock import Mock
import pytest
from airflow.providers.openai.operators.openai import OpenAIEmbeddingOperator
from airflow.utils.context import Context

def test_execute_with_input_text():
    if False:
        while True:
            i = 10
    operator = OpenAIEmbeddingOperator(task_id='TaskId', conn_id='test_conn_id', model='test_model', input_text='Test input text')
    mock_hook_instance = Mock()
    mock_hook_instance.create_embeddings.return_value = [1.0, 2.0, 3.0]
    operator.hook = mock_hook_instance
    context = Context()
    embeddings = operator.execute(context)
    assert embeddings == [1.0, 2.0, 3.0]

@pytest.mark.parametrize('invalid_input', ['', None, 123])
def test_execute_with_invalid_input(invalid_input):
    if False:
        return 10
    with pytest.raises(ValueError):
        operator = OpenAIEmbeddingOperator(task_id='TaskId', conn_id='test_conn_id', model='test_model', input_text=invalid_input)
        context = Context()
        operator.execute(context)