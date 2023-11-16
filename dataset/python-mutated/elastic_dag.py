from __future__ import annotations
import enum
import os
import re
from datetime import datetime, timedelta
from enum import Enum
from airflow.models.baseoperator import chain
from airflow.models.dag import DAG
from airflow.operators.bash import BashOperator
RE_TIME_DELTA = re.compile('^((?P<days>[.\\d]+?)d)?((?P<hours>[.\\d]+?)h)?((?P<minutes>[.\\d]+?)m)?((?P<seconds>[.\\d]+?)s)?$')

def parse_time_delta(time_str: str):
    if False:
        print('Hello World!')
    '\n    Parse a time string e.g. (2h13m) into a timedelta object.\n\n    :param time_str: A string identifying a duration.  (eg. 2h13m)\n    :return datetime.timedelta: A datetime.timedelta object or "@once"\n    '
    parts = RE_TIME_DELTA.match(time_str)
    assert parts is not None, f"Could not parse any time information from '{time_str}'. Examples of valid strings: '8h', '2d8h5m20s', '2m4s'"
    time_params = {name: float(param) for (name, param) in parts.groupdict().items() if param}
    return timedelta(**time_params)

def parse_schedule_interval(time_str: str):
    if False:
        for i in range(10):
            print('nop')
    '\n    Parse a schedule interval string e.g. (2h13m) or "@once".\n\n    :param time_str: A string identifying a schedule interval.  (eg. 2h13m, None, @once)\n    :return datetime.timedelta: A datetime.timedelta object or "@once" or None\n    '
    if time_str == 'None':
        return None
    if time_str == '@once':
        return '@once'
    return parse_time_delta(time_str)

def safe_dag_id(s: str) -> str:
    if False:
        for i in range(10):
            print('nop')
    '\n    Remove invalid characters for dag_id\n    '
    return re.sub('[^0-9a-zA-Z_]+', '_', s)

def chain_as_binary_tree(*tasks: BashOperator):
    if False:
        while True:
            i = 10
    '\n    Chain tasks as a binary tree where task i is child of task (i - 1) // 2 :\n\n        t0 -> t1 -> t3 -> t7\n          |    \\\n          |      -> t4 -> t8\n          |\n           -> t2 -> t5 -> t9\n               \\\n                 -> t6\n    '
    for i in range(1, len(tasks)):
        tasks[i].set_downstream(tasks[(i - 1) // 2])

def chain_as_grid(*tasks: BashOperator):
    if False:
        for i in range(10):
            print('nop')
    '\n    Chain tasks as a grid:\n\n     t0 -> t1 -> t2 -> t3\n      |     |     |\n      v     v     v\n     t4 -> t5 -> t6\n      |     |\n      v     v\n     t7 -> t8\n      |\n      v\n     t9\n    '
    if len(tasks) > 100 * 99 / 2:
        raise ValueError('Cannot generate grid DAGs with lateral size larger than 100 tasks.')
    grid_size = next((n for n in range(100) if n * (n + 1) / 2 >= len(tasks)))

    def index(i, j):
        if False:
            while True:
                i = 10
        '\n        Return the index of node (i, j) on the grid.\n        '
        return int(grid_size * i - i * (i - 1) / 2 + j)
    for i in range(grid_size - 1):
        for j in range(grid_size - i - 1):
            if index(i + 1, j) < len(tasks):
                tasks[index(i + 1, j)].set_downstream(tasks[index(i, j)])
            if index(i, j + 1) < len(tasks):
                tasks[index(i, j + 1)].set_downstream(tasks[index(i, j)])

def chain_as_star(*tasks: BashOperator):
    if False:
        return 10
    '\n    Chain tasks as a star (all tasks are children of task 0)\n\n     t0 -> t1\n      | -> t2\n      | -> t3\n      | -> t4\n      | -> t5\n    '
    tasks[0].set_upstream(list(tasks[1:]))

@enum.unique
class DagShape(Enum):
    """
    Define shape of the Dag that will be used for testing.
    """
    NO_STRUCTURE = 'no_structure'
    LINEAR = 'linear'
    BINARY_TREE = 'binary_tree'
    STAR = 'star'
    GRID = 'grid'
DAG_PREFIX = os.environ.get('PERF_DAG_PREFIX', 'perf_scheduler')
DAG_COUNT = int(os.environ['PERF_DAGS_COUNT'])
TASKS_COUNT = int(os.environ['PERF_TASKS_COUNT'])
START_DATE_ENV = os.environ.get('PERF_START_AGO', '1h')
START_DATE = datetime.now() - parse_time_delta(START_DATE_ENV)
SCHEDULE_INTERVAL_ENV = os.environ.get('PERF_SCHEDULE_INTERVAL', '@once')
SCHEDULE_INTERVAL = parse_schedule_interval(SCHEDULE_INTERVAL_ENV)
SHAPE = DagShape(os.environ['PERF_SHAPE'])
args = {'owner': 'airflow', 'start_date': START_DATE}
if 'PERF_MAX_RUNS' in os.environ:
    if isinstance(SCHEDULE_INTERVAL, str):
        raise ValueError("Can't set max runs with string-based schedule_interval")
    num_runs = int(os.environ['PERF_MAX_RUNS'])
    args['end_date'] = START_DATE + SCHEDULE_INTERVAL * (num_runs - 1)
for dag_no in range(1, DAG_COUNT + 1):
    dag = DAG(dag_id=safe_dag_id('__'.join([DAG_PREFIX, f'SHAPE={SHAPE.name.lower()}', f'DAGS_COUNT={dag_no}_of_{DAG_COUNT}', f'TASKS_COUNT=${TASKS_COUNT}', f'START_DATE=${START_DATE_ENV}', f'SCHEDULE=${SCHEDULE_INTERVAL_ENV}'])), is_paused_upon_creation=False, default_args=args, schedule=SCHEDULE_INTERVAL)
    elastic_dag_tasks = [BashOperator(task_id='__'.join(['tasks', f'{i}_of_{TASKS_COUNT}']), bash_command='echo test', dag=dag) for i in range(1, TASKS_COUNT + 1)]
    shape_function_map = {DagShape.LINEAR: chain, DagShape.BINARY_TREE: chain_as_binary_tree, DagShape.STAR: chain_as_star, DagShape.GRID: chain_as_grid}
    if SHAPE != DagShape.NO_STRUCTURE:
        shape_function_map[SHAPE](*elastic_dag_tasks)
    globals()[f'dag_{dag_no}'] = dag