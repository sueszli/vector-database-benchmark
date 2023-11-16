from __future__ import annotations
import datetime
from airflow.models.dag import DAG
with DAG(dag_id='test_mapped_taskflow', start_date=datetime.datetime(2022, 1, 1)) as dag:

    @dag.task
    def make_list():
        if False:
            print('Hello World!')
        return [1, 2, {'a': 'b'}]

    @dag.task
    def consumer(value):
        if False:
            for i in range(10):
                print('nop')
        print(repr(value))
    consumer.expand(value=make_list())
    consumer.expand(value=[1, 2, 3])