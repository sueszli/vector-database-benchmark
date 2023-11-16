"""
This is an example DAG which uses the KylinCubeOperator.
The tasks below include kylin build, refresh, merge operation.
"""
from __future__ import annotations
import os
from datetime import datetime
from airflow import DAG
from airflow.providers.apache.kylin.operators.kylin_cube import KylinCubeOperator
ENV_ID = os.environ.get('SYSTEM_TESTS_ENV_ID')
DAG_ID = 'example_kylin_operator'
with DAG(dag_id=DAG_ID, schedule=None, start_date=datetime(2021, 1, 1), catchup=False, default_args={'project': 'learn_kylin', 'cube': 'kylin_sales_cube'}, tags=['example']) as dag:

    @dag.task
    def gen_build_time():
        if False:
            for i in range(10):
                print('nop')
        '\n        Gen build time and push to XCom (with key of "return_value")\n        :return: A dict with build time values.\n        '
        return {'date_start': '1325347200000', 'date_end': '1325433600000'}
    gen_build_time_task = gen_build_time()
    gen_build_time_output_date_start = gen_build_time_task['date_start']
    gen_build_time_output_date_end = gen_build_time_task['date_end']
    build_task1 = KylinCubeOperator(task_id='kylin_build_1', command='build', start_time=gen_build_time_output_date_start, end_time=gen_build_time_output_date_end, is_track_job=True)
    build_task2 = KylinCubeOperator(task_id='kylin_build_2', command='build', start_time=gen_build_time_output_date_end, end_time='1325520000000', is_track_job=True)
    refresh_task1 = KylinCubeOperator(task_id='kylin_refresh_1', command='refresh', start_time=gen_build_time_output_date_start, end_time=gen_build_time_output_date_end, is_track_job=True)
    merge_task = KylinCubeOperator(task_id='kylin_merge', command='merge', start_time=gen_build_time_output_date_start, end_time='1325520000000', is_track_job=True)
    disable_task = KylinCubeOperator(task_id='kylin_disable', command='disable')
    purge_task = KylinCubeOperator(task_id='kylin_purge', command='purge')
    build_task3 = KylinCubeOperator(task_id='kylin_build_3', command='build', start_time=gen_build_time_output_date_end, end_time='1328730000000')
    build_task1 >> build_task2 >> refresh_task1 >> merge_task >> disable_task >> purge_task >> build_task3
    from tests.system.utils.watcher import watcher
    list(dag.tasks) >> watcher()
from tests.system.utils import get_test_run
test_run = get_test_run(dag)