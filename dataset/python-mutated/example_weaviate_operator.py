from __future__ import annotations
import pendulum
from airflow.decorators import dag, task, teardown
from airflow.providers.weaviate.hooks.weaviate import WeaviateHook
from airflow.providers.weaviate.operators.weaviate import WeaviateIngestOperator
sample_data_with_vector = [{'Answer': 'Liver', 'Category': 'SCIENCE', 'Question': 'This organ removes excess glucose from the blood & stores it as glycogen', 'Vector': [-0.006632288, -0.0042016874, 0.030541966]}, {'Answer': 'Elephant', 'Category': 'ANIMALS', 'Question': "It's the only living mammal in the order Proboseidea", 'Vector': [-0.0166891, -0.00092290324, -0.0125168245]}, {'Answer': 'the nose or snout', 'Category': 'ANIMALS', 'Question': 'The gavial looks very much like a crocodile except for this bodily feature', 'Vector': [-0.015592773, 0.019883318, 0.017782344]}]
sample_data_without_vector = [{'Answer': 'Liver', 'Category': 'SCIENCE', 'Question': 'This organ removes excess glucose from the blood & stores it as glycogen'}, {'Answer': 'Elephant', 'Category': 'ANIMALS', 'Question': "It's the only living mammal in the order Proboseidea"}, {'Answer': 'the nose or snout', 'Category': 'ANIMALS', 'Question': 'The gavial looks very much like a crocodile except for this bodily feature'}]

def get_data_with_vectors(*args, **kwargs):
    if False:
        return 10
    return sample_data_with_vector

def get_data_without_vectors(*args, **kwargs):
    if False:
        i = 10
        return i + 15
    return sample_data_without_vector

@dag(schedule=None, start_date=pendulum.datetime(2021, 1, 1, tz='UTC'), catchup=False, tags=['example', 'weaviate'])
def example_weaviate_using_operator():
    if False:
        print('Hello World!')
    '\n    Example Weaviate DAG demonstrating usage of the operator.\n    '

    @task()
    def create_class_without_vectorizer():
        if False:
            for i in range(10):
                print('nop')
        "\n        Example task to create class without any Vectorizer. You're expected to provide custom vectors\n         for your data.\n        "
        weaviate_hook = WeaviateHook()
        class_obj = {'class': 'QuestionWithoutVectorizerUsingOperator', 'vectorizer': 'none'}
        weaviate_hook.create_class(class_obj)

    @task(trigger_rule='all_done')
    def store_data_with_vectors_in_xcom():
        if False:
            print('Hello World!')
        return sample_data_with_vector
    batch_data_with_vectors_xcom_data = WeaviateIngestOperator(task_id='batch_data_with_vectors_xcom_data', conn_id='weaviate_default', class_name='QuestionWithoutVectorizerUsingOperator', input_json=store_data_with_vectors_in_xcom(), trigger_rule='all_done')
    batch_data_with_vectors_callable_data = WeaviateIngestOperator(task_id='batch_data_with_vectors_callable_data', conn_id='weaviate_default', class_name='QuestionWithoutVectorizerUsingOperator', input_json=get_data_with_vectors(), trigger_rule='all_done')

    @task()
    def create_class_with_vectorizer():
        if False:
            i = 10
            return i + 15
        '\n        Example task to create class with OpenAI Vectorizer responsible for vectorining data using Weaviate\n         cluster.\n        '
        weaviate_hook = WeaviateHook()
        class_obj = {'class': 'QuestionWithOpenAIVectorizerUsingOperator', 'description': 'Information from a Jeopardy! question', 'properties': [{'dataType': ['text'], 'description': 'The question', 'name': 'question'}, {'dataType': ['text'], 'description': 'The answer', 'name': 'answer'}, {'dataType': ['text'], 'description': 'The category', 'name': 'category'}], 'vectorizer': 'text2vec-openai'}
        weaviate_hook.create_class(class_obj)

    @task(trigger_rule='all_done')
    def store_data_without_vectors_in_xcom():
        if False:
            i = 10
            return i + 15
        return sample_data_without_vector
    batch_data_without_vectors_xcom_data = WeaviateIngestOperator(task_id='batch_data_without_vectors_xcom_data', conn_id='weaviate_default', class_name='QuestionWithOpenAIVectorizerUsingOperator', input_json=store_data_without_vectors_in_xcom(), trigger_rule='all_done')
    batch_data_without_vectors_callable_data = WeaviateIngestOperator(task_id='batch_data_without_vectors_callable_data', conn_id='weaviate_default', class_name='QuestionWithOpenAIVectorizerUsingOperator', input_json=get_data_without_vectors(), trigger_rule='all_done')

    @teardown
    @task
    def delete_weaviate_class_Vector():
        if False:
            return 10
        '\n        Example task to delete a weaviate class\n        '
        weaviate_hook = WeaviateHook()
        weaviate_hook.delete_class('QuestionWithOpenAIVectorizerUsingOperator')

    @teardown
    @task
    def delete_weaviate_class_without_Vector():
        if False:
            print('Hello World!')
        '\n        Example task to delete a weaviate class\n        '
        weaviate_hook = WeaviateHook()
        weaviate_hook.delete_class('QuestionWithoutVectorizerUsingOperator')
    create_class_without_vectorizer() >> [batch_data_with_vectors_xcom_data, batch_data_with_vectors_callable_data] >> delete_weaviate_class_without_Vector()
    create_class_with_vectorizer() >> [batch_data_without_vectors_xcom_data, batch_data_without_vectors_callable_data] >> delete_weaviate_class_Vector()
example_weaviate_using_operator()
from tests.system.utils import get_test_run
test_run = get_test_run(dag)