from __future__ import annotations
import warnings
from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Any, Iterable, Sequence
from airflow.exceptions import AirflowException, RemovedInAirflow3Warning
from airflow.utils.types import NOTSET
if TYPE_CHECKING:
    from logging import Logger
    import pendulum
    from airflow.models.baseoperator import BaseOperator
    from airflow.models.dag import DAG
    from airflow.models.operator import Operator
    from airflow.serialization.enums import DagAttributeTypes
    from airflow.utils.edgemodifier import EdgeModifier
    from airflow.utils.task_group import TaskGroup
    from airflow.utils.types import ArgNotSet

class DependencyMixin:
    """Mixing implementing common dependency setting methods like >> and <<."""

    @property
    def roots(self) -> Sequence[DependencyMixin]:
        if False:
            while True:
                i = 10
        '\n        List of root nodes -- ones with no upstream dependencies.\n\n        a.k.a. the "start" of this sub-graph\n        '
        raise NotImplementedError()

    @property
    def leaves(self) -> Sequence[DependencyMixin]:
        if False:
            i = 10
            return i + 15
        '\n        List of leaf nodes -- ones with only upstream dependencies.\n\n        a.k.a. the "end" of this sub-graph\n        '
        raise NotImplementedError()

    @abstractmethod
    def set_upstream(self, other: DependencyMixin | Sequence[DependencyMixin], edge_modifier: EdgeModifier | None=None):
        if False:
            for i in range(10):
                print('nop')
        'Set a task or a task list to be directly upstream from the current task.'
        raise NotImplementedError()

    @abstractmethod
    def set_downstream(self, other: DependencyMixin | Sequence[DependencyMixin], edge_modifier: EdgeModifier | None=None):
        if False:
            for i in range(10):
                print('nop')
        'Set a task or a task list to be directly downstream from the current task.'
        raise NotImplementedError()

    def as_setup(self) -> DependencyMixin:
        if False:
            while True:
                i = 10
        'Mark a task as setup task.'
        raise NotImplementedError()

    def as_teardown(self, *, setups: BaseOperator | Iterable[BaseOperator] | ArgNotSet=NOTSET, on_failure_fail_dagrun=NOTSET) -> DependencyMixin:
        if False:
            while True:
                i = 10
        'Mark a task as teardown and set its setups as direct relatives.'
        raise NotImplementedError()

    def update_relative(self, other: DependencyMixin, upstream: bool=True, edge_modifier: EdgeModifier | None=None) -> None:
        if False:
            while True:
                i = 10
        '\n        Update relationship information about another TaskMixin. Default is no-op.\n\n        Override if necessary.\n        '

    def __lshift__(self, other: DependencyMixin | Sequence[DependencyMixin]):
        if False:
            return 10
        'Implement Task << Task.'
        self.set_upstream(other)
        return other

    def __rshift__(self, other: DependencyMixin | Sequence[DependencyMixin]):
        if False:
            for i in range(10):
                print('nop')
        'Implement Task >> Task.'
        self.set_downstream(other)
        return other

    def __rrshift__(self, other: DependencyMixin | Sequence[DependencyMixin]):
        if False:
            for i in range(10):
                print('nop')
        "Implement Task >> [Task] because list don't have __rshift__ operators."
        self.__lshift__(other)
        return self

    def __rlshift__(self, other: DependencyMixin | Sequence[DependencyMixin]):
        if False:
            print('Hello World!')
        "Implement Task << [Task] because list don't have __lshift__ operators."
        self.__rshift__(other)
        return self

    @classmethod
    def _iter_references(cls, obj: Any) -> Iterable[tuple[DependencyMixin, str]]:
        if False:
            while True:
                i = 10
        from airflow.models.baseoperator import AbstractOperator
        from airflow.utils.mixins import ResolveMixin
        if isinstance(obj, AbstractOperator):
            yield (obj, 'operator')
        elif isinstance(obj, ResolveMixin):
            yield from obj.iter_references()
        elif isinstance(obj, Sequence):
            for o in obj:
                yield from cls._iter_references(o)

class TaskMixin(DependencyMixin):
    """Mixin to provide task-related things.

    :meta private:
    """

    def __init_subclass__(cls) -> None:
        if False:
            for i in range(10):
                print('nop')
        warnings.warn(f'TaskMixin has been renamed to DependencyMixin, please update {cls.__name__}', category=RemovedInAirflow3Warning, stacklevel=2)
        return super().__init_subclass__()

class DAGNode(DependencyMixin, metaclass=ABCMeta):
    """
    A base class for a node in the graph of a workflow.

    A node may be an Operator or a Task Group, either mapped or unmapped.
    """
    dag: DAG | None = None
    task_group: TaskGroup | None = None
    'The task_group that contains this node'

    @property
    @abstractmethod
    def node_id(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError()

    @property
    def label(self) -> str | None:
        if False:
            return 10
        tg = self.task_group
        if tg and tg.node_id and tg.prefix_group_id:
            return self.node_id[len(tg.node_id) + 1:]
        return self.node_id
    start_date: pendulum.DateTime | None
    end_date: pendulum.DateTime | None
    upstream_task_ids: set[str]
    downstream_task_ids: set[str]

    def has_dag(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return self.dag is not None

    @property
    def dag_id(self) -> str:
        if False:
            i = 10
            return i + 15
        'Returns dag id if it has one or an adhoc/meaningless ID.'
        if self.dag:
            return self.dag.dag_id
        return '_in_memory_dag_'

    @property
    def log(self) -> Logger:
        if False:
            return 10
        raise NotImplementedError()

    @property
    @abstractmethod
    def roots(self) -> Sequence[DAGNode]:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError()

    @property
    @abstractmethod
    def leaves(self) -> Sequence[DAGNode]:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError()

    def _set_relatives(self, task_or_task_list: DependencyMixin | Sequence[DependencyMixin], upstream: bool=False, edge_modifier: EdgeModifier | None=None) -> None:
        if False:
            i = 10
            return i + 15
        'Set relatives for the task or task list.'
        from airflow.models.baseoperator import BaseOperator
        from airflow.models.mappedoperator import MappedOperator
        if not isinstance(task_or_task_list, Sequence):
            task_or_task_list = [task_or_task_list]
        task_list: list[Operator] = []
        for task_object in task_or_task_list:
            task_object.update_relative(self, not upstream, edge_modifier=edge_modifier)
            relatives = task_object.leaves if upstream else task_object.roots
            for task in relatives:
                if not isinstance(task, (BaseOperator, MappedOperator)):
                    raise AirflowException(f'Relationships can only be set between Operators; received {task.__class__.__name__}')
                task_list.append(task)
        dags: set[DAG] = {task.dag for task in [*self.roots, *task_list] if task.has_dag() and task.dag}
        if len(dags) > 1:
            raise AirflowException(f'Tried to set relationships between tasks in more than one DAG: {dags}')
        elif len(dags) == 1:
            dag = dags.pop()
        else:
            raise AirflowException(f"Tried to create relationships between tasks that don't have DAGs yet. Set the DAG for at least one task and try again: {[self, *task_list]}")
        if not self.has_dag():
            self.dag = dag
        for task in task_list:
            if dag and (not task.has_dag()):
                dag.add_task(task)
            if upstream:
                task.downstream_task_ids.add(self.node_id)
                self.upstream_task_ids.add(task.node_id)
                if edge_modifier:
                    edge_modifier.add_edge_info(self.dag, task.node_id, self.node_id)
            else:
                self.downstream_task_ids.add(task.node_id)
                task.upstream_task_ids.add(self.node_id)
                if edge_modifier:
                    edge_modifier.add_edge_info(self.dag, self.node_id, task.node_id)

    def set_downstream(self, task_or_task_list: DependencyMixin | Sequence[DependencyMixin], edge_modifier: EdgeModifier | None=None) -> None:
        if False:
            while True:
                i = 10
        'Set a node (or nodes) to be directly downstream from the current node.'
        self._set_relatives(task_or_task_list, upstream=False, edge_modifier=edge_modifier)

    def set_upstream(self, task_or_task_list: DependencyMixin | Sequence[DependencyMixin], edge_modifier: EdgeModifier | None=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Set a node (or nodes) to be directly upstream from the current node.'
        self._set_relatives(task_or_task_list, upstream=True, edge_modifier=edge_modifier)

    @property
    def downstream_list(self) -> Iterable[Operator]:
        if False:
            i = 10
            return i + 15
        'List of nodes directly downstream.'
        if not self.dag:
            raise AirflowException(f'Operator {self} has not been assigned to a DAG yet')
        return [self.dag.get_task(tid) for tid in self.downstream_task_ids]

    @property
    def upstream_list(self) -> Iterable[Operator]:
        if False:
            return 10
        'List of nodes directly upstream.'
        if not self.dag:
            raise AirflowException(f'Operator {self} has not been assigned to a DAG yet')
        return [self.dag.get_task(tid) for tid in self.upstream_task_ids]

    def get_direct_relative_ids(self, upstream: bool=False) -> set[str]:
        if False:
            i = 10
            return i + 15
        'Get set of the direct relative ids to the current task, upstream or downstream.'
        if upstream:
            return self.upstream_task_ids
        else:
            return self.downstream_task_ids

    def get_direct_relatives(self, upstream: bool=False) -> Iterable[DAGNode]:
        if False:
            return 10
        'Get list of the direct relatives to the current task, upstream or downstream.'
        if upstream:
            return self.upstream_list
        else:
            return self.downstream_list

    def serialize_for_task_group(self) -> tuple[DagAttributeTypes, Any]:
        if False:
            i = 10
            return i + 15
        "Serialize a task group's content; used by TaskGroupSerialization."
        raise NotImplementedError()