from __future__ import annotations
from typing import TYPE_CHECKING
from airflow.models.taskmap import TaskMap
if TYPE_CHECKING:
    from sqlalchemy.orm import Session
    from airflow.models.mappedoperator import MappedOperator

def expand_mapped_task(mapped: MappedOperator, run_id: str, upstream_task_id: str, length: int, session: Session):
    if False:
        print('Hello World!')
    session.add(TaskMap(dag_id=mapped.dag_id, task_id=upstream_task_id, run_id=run_id, map_index=-1, length=length, keys=None))
    session.flush()
    mapped.expand_mapped_task(run_id, session=session)