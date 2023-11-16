from __future__ import annotations
from argparse import ArgumentError
COMMAND_MAP = {'worker': 'celery worker', 'flower': 'celery flower', 'trigger_dag': 'dags trigger', 'delete_dag': 'dags delete', 'show_dag': 'dags show', 'list_dag': 'dags list', 'dag_status': 'dags status', 'backfill': 'dags backfill', 'list_dag_runs': 'dags list-runs', 'pause': 'dags pause', 'unpause': 'dags unpause', 'test': 'tasks test', 'clear': 'tasks clear', 'list_tasks': 'tasks list', 'task_failed_deps': 'tasks failed-deps', 'task_state': 'tasks state', 'run': 'tasks run', 'render': 'tasks render', 'initdb': 'db init', 'resetdb': 'db reset', 'upgradedb': 'db upgrade', 'checkdb': 'db check', 'shell': 'db shell', 'pool': 'pools', 'list_users': 'users list', 'create_user': 'users create', 'delete_user': 'users delete'}

def check_legacy_command(action, value):
    if False:
        while True:
            i = 10
    'Check command value and raise error if value is in removed command.'
    new_command = COMMAND_MAP.get(value)
    if new_command is not None:
        msg = f'`airflow {value}` command, has been removed, please use `airflow {new_command}`'
        raise ArgumentError(action, msg)