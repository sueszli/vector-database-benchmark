"""DAG Cycle tester."""
from __future__ import annotations
from collections import defaultdict, deque
from typing import TYPE_CHECKING
from airflow.exceptions import AirflowDagCycleException, RemovedInAirflow3Warning
if TYPE_CHECKING:
    from airflow.models.dag import DAG
CYCLE_NEW = 0
CYCLE_IN_PROGRESS = 1
CYCLE_DONE = 2

def test_cycle(dag: DAG) -> None:
    if False:
        print('Hello World!')
    "\n    A wrapper function of `check_cycle` for backward compatibility purpose.\n\n    New code should use `check_cycle` instead since this function name `test_cycle` starts\n    with 'test_' and will be considered as a unit test by pytest, resulting in failure.\n    "
    from warnings import warn
    warn('Deprecated, please use `check_cycle` at the same module instead.', RemovedInAirflow3Warning, stacklevel=2)
    return check_cycle(dag)

def check_cycle(dag: DAG) -> None:
    if False:
        i = 10
        return i + 15
    'Check to see if there are any cycles in the DAG.\n\n    :raises AirflowDagCycleException: If cycle is found in the DAG.\n    '
    visited: dict[str, int] = defaultdict(int)
    path_stack: deque[str] = deque()
    task_dict = dag.task_dict

    def _check_adjacent_tasks(task_id, current_task):
        if False:
            while True:
                i = 10
        'Return first untraversed child task, else None if all tasks traversed.'
        for adjacent_task in current_task.get_direct_relative_ids():
            if visited[adjacent_task] == CYCLE_IN_PROGRESS:
                msg = f'Cycle detected in DAG: {dag.dag_id}. Faulty task: {task_id}'
                raise AirflowDagCycleException(msg)
            elif visited[adjacent_task] == CYCLE_NEW:
                return adjacent_task
        return None
    for dag_task_id in dag.task_dict.keys():
        if visited[dag_task_id] == CYCLE_DONE:
            continue
        path_stack.append(dag_task_id)
        while path_stack:
            current_task_id = path_stack[-1]
            if visited[current_task_id] == CYCLE_NEW:
                visited[current_task_id] = CYCLE_IN_PROGRESS
            task = task_dict[current_task_id]
            child_to_check = _check_adjacent_tasks(current_task_id, task)
            if not child_to_check:
                visited[current_task_id] = CYCLE_DONE
                path_stack.pop()
            else:
                path_stack.append(child_to_check)