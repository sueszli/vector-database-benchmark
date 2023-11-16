from __future__ import annotations
from datetime import datetime, timedelta
from opensearchpy import Integer, Text
from opensearchpy.helpers.document import Document
from opensearchpy.helpers.search import Search
from airflow.models.baseoperator import chain
from airflow.models.dag import DAG
from airflow.providers.opensearch.operators.opensearch import OpenSearchAddDocumentOperator, OpenSearchCreateIndexOperator, OpenSearchQueryOperator
DAG_ID = 'example_opensearch'
INDEX_NAME = 'example_index'
default_args = {'owner': 'airflow', 'depend_on_past': False, 'email_on_failure': False, 'email_on_retry': False, 'retries': 1, 'retry_delay': timedelta(minutes=5)}

class LogDocument(Document):
    log_group_id = Integer()
    logger = Text()
    message = Text()

    class Index:
        name = INDEX_NAME

    def save(self, **kwargs):
        if False:
            while True:
                i = 10
        super().save(**kwargs)

def load_connections():
    if False:
        return 10
    from airflow.models import Connection
    from airflow.utils import db
    db.merge_conn(Connection(conn_id='opensearch_test', conn_type='opensearch', host='127.0.0.1', login='test', password='test'))
with DAG(dag_id=DAG_ID, start_date=datetime(2021, 1, 1), schedule='@once', catchup=False, tags=['example'], default_args=default_args, description='Examples of OpenSearch Operators') as dag:
    create_index = OpenSearchCreateIndexOperator(task_id='create_index', index_name=INDEX_NAME, index_body={'settings': {'index': {'number_of_shards': 1}}})
    add_document_by_args = OpenSearchAddDocumentOperator(task_id='add_document_with_args', index_name=INDEX_NAME, doc_id=1, document={'log_group_id': 1, 'logger': 'python', 'message': 'hello world'})
    add_document_by_class = OpenSearchAddDocumentOperator(task_id='add_document_by_class', doc_class=LogDocument(log_group_id=2, logger='airflow', message='hello airflow'))
    search_low_level = OpenSearchQueryOperator(task_id='low_level_query', index_name='system_test', query={'query': {'bool': {'must': {'match': {'message': 'hello world'}}}}})
    search = Search()
    search.index = INDEX_NAME
    search_object = search.filter('term', logger='airflow').query('match', message='hello airflow')
    search_high_level = OpenSearchQueryOperator(task_id='high_level_query', search_object=search_object)
    chain(create_index, add_document_by_class, add_document_by_args, search_high_level, search_low_level)
    from tests.system.utils.watcher import watcher
    list(dag.tasks) >> watcher()
from tests.system.utils import get_test_run
test_run = get_test_run(dag)