from __future__ import annotations
import pendulum
from airflow.decorators import dag, task
from airflow.providers.openai.hooks.openai import OpenAIHook
from airflow.providers.openai.operators.openai import OpenAIEmbeddingOperator

def input_text_callable(input_arg1: str, input_arg2: str, input_kwarg1: str='default_kwarg1_value', input_kwarg2: str='default_kwarg1_value'):
    if False:
        i = 10
        return i + 15
    text = ' '.join([input_arg1, input_arg2, input_kwarg1, input_kwarg2])
    return text

@dag(schedule=None, start_date=pendulum.datetime(2021, 1, 1, tz='UTC'), catchup=False, tags=['example', 'openai'])
def example_openai_dag():
    if False:
        for i in range(10):
            print('nop')
    '\n    ### TaskFlow API Tutorial Documentation\n    This is a simple data pipeline example which demonstrates the use of\n    the TaskFlow API using three simple tasks for Extract, Transform, and Load.\n    Documentation that goes along with the Airflow TaskFlow API tutorial is\n    located\n    [here](https://airflow.apache.org/docs/apache-airflow/stable/tutorial_taskflow_api.html)\n    '
    texts = ['On Kernel-Target Alignment. We describe a family of global optimization procedures', ' that automatically decompose optimization problems into smaller loosely coupled', ' problems, then combine the solutions of these with message passing algorithms.']

    @task()
    def create_embeddings_using_hook():
        if False:
            return 10
        '\n        #### Extract task\n        A simple Extract task to get data ready for the rest of the data\n        pipeline. In this case, getting data is simulated by reading from a\n        hardcoded JSON string.\n        '
        openai_hook = OpenAIHook()
        embeddings = openai_hook.create_embeddings(texts[0])
        return embeddings

    @task()
    def task_to_store_input_text_in_xcom():
        if False:
            while True:
                i = 10
        return texts[0]
    OpenAIEmbeddingOperator(task_id='embedding_using_xcom_data', conn_id='openai_default', input_text=task_to_store_input_text_in_xcom(), model='text-embedding-ada-002')
    OpenAIEmbeddingOperator(task_id='embedding_using_callable', conn_id='openai_default', input_text=input_text_callable('input_arg1_value', 'input2_value', input_kwarg1='input_kwarg1_value', input_kwarg2='input_kwarg2_value'), model='text-embedding-ada-002')
    OpenAIEmbeddingOperator(task_id='embedding_using_text', conn_id='openai_default', input_text=texts, model='text-embedding-ada-002')
    create_embeddings_using_hook()
example_openai_dag()
from tests.system.utils import get_test_run
test_run = get_test_run(dag)