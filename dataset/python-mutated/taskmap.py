"""Table to store information about mapped task instances (AIP-42)."""
from __future__ import annotations
import collections.abc
import enum
from typing import TYPE_CHECKING, Any, Collection
from sqlalchemy import CheckConstraint, Column, ForeignKeyConstraint, Integer, String
from airflow.models.base import COLLATION_ARGS, ID_LEN, Base
from airflow.utils.sqlalchemy import ExtendedJSON
if TYPE_CHECKING:
    from airflow.models.taskinstance import TaskInstance
    from airflow.serialization.pydantic.taskinstance import TaskInstancePydantic

class TaskMapVariant(enum.Enum):
    """Task map variant.

    Possible values are **dict** (for a key-value mapping) and **list** (for an
    ordered value sequence).
    """
    DICT = 'dict'
    LIST = 'list'

class TaskMap(Base):
    """Model to track dynamic task-mapping information.

    This is currently only populated by an upstream TaskInstance pushing an
    XCom that's pulled by a downstream for mapping purposes.
    """
    __tablename__ = 'task_map'
    dag_id = Column(String(ID_LEN, **COLLATION_ARGS), primary_key=True)
    task_id = Column(String(ID_LEN, **COLLATION_ARGS), primary_key=True)
    run_id = Column(String(ID_LEN, **COLLATION_ARGS), primary_key=True)
    map_index = Column(Integer, primary_key=True)
    length = Column(Integer, nullable=False)
    keys = Column(ExtendedJSON, nullable=True)
    __table_args__ = (CheckConstraint(length >= 0, name='task_map_length_not_negative'), ForeignKeyConstraint([dag_id, task_id, run_id, map_index], ['task_instance.dag_id', 'task_instance.task_id', 'task_instance.run_id', 'task_instance.map_index'], name='task_map_task_instance_fkey', ondelete='CASCADE', onupdate='CASCADE'))

    def __init__(self, dag_id: str, task_id: str, run_id: str, map_index: int, length: int, keys: list[Any] | None) -> None:
        if False:
            i = 10
            return i + 15
        self.dag_id = dag_id
        self.task_id = task_id
        self.run_id = run_id
        self.map_index = map_index
        self.length = length
        self.keys = keys

    @classmethod
    def from_task_instance_xcom(cls, ti: TaskInstance | TaskInstancePydantic, value: Collection) -> TaskMap:
        if False:
            i = 10
            return i + 15
        if ti.run_id is None:
            raise ValueError('cannot record task map for unrun task instance')
        return cls(dag_id=ti.dag_id, task_id=ti.task_id, run_id=ti.run_id, map_index=ti.map_index, length=len(value), keys=list(value) if isinstance(value, collections.abc.Mapping) else None)

    @property
    def variant(self) -> TaskMapVariant:
        if False:
            print('Hello World!')
        if self.keys is None:
            return TaskMapVariant.LIST
        return TaskMapVariant.DICT