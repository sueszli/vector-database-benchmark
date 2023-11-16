from __future__ import annotations
import datetime
import inspect
from functools import cached_property
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Collection, Iterable, Iterator, Sequence
from sqlalchemy import select
from airflow.compat.functools import cache
from airflow.configuration import conf
from airflow.exceptions import AirflowException
from airflow.models.expandinput import NotFullyPopulated
from airflow.models.taskmixin import DAGNode, DependencyMixin
from airflow.template.templater import Templater
from airflow.utils.context import Context
from airflow.utils.db import exists_query
from airflow.utils.log.secrets_masker import redact
from airflow.utils.session import NEW_SESSION, provide_session
from airflow.utils.setup_teardown import SetupTeardownContext
from airflow.utils.sqlalchemy import skip_locked, with_row_locks
from airflow.utils.state import State, TaskInstanceState
from airflow.utils.task_group import MappedTaskGroup
from airflow.utils.trigger_rule import TriggerRule
from airflow.utils.types import NOTSET, ArgNotSet
from airflow.utils.weight_rule import WeightRule
TaskStateChangeCallback = Callable[[Context], None]
if TYPE_CHECKING:
    import jinja2
    from sqlalchemy.orm import Session
    from airflow.models.baseoperator import BaseOperator, BaseOperatorLink
    from airflow.models.dag import DAG
    from airflow.models.mappedoperator import MappedOperator
    from airflow.models.operator import Operator
    from airflow.models.taskinstance import TaskInstance
    from airflow.utils.task_group import TaskGroup
DEFAULT_OWNER: str = conf.get_mandatory_value('operators', 'default_owner')
DEFAULT_POOL_SLOTS: int = 1
DEFAULT_PRIORITY_WEIGHT: int = 1
DEFAULT_QUEUE: str = conf.get_mandatory_value('operators', 'default_queue')
DEFAULT_IGNORE_FIRST_DEPENDS_ON_PAST: bool = conf.getboolean('scheduler', 'ignore_first_depends_on_past_by_default')
DEFAULT_WAIT_FOR_PAST_DEPENDS_BEFORE_SKIPPING: bool = False
DEFAULT_RETRIES: int = conf.getint('core', 'default_task_retries', fallback=0)
DEFAULT_RETRY_DELAY: datetime.timedelta = datetime.timedelta(seconds=conf.getint('core', 'default_task_retry_delay', fallback=300))
MAX_RETRY_DELAY: int = conf.getint('core', 'max_task_retry_delay', fallback=24 * 60 * 60)
DEFAULT_WEIGHT_RULE: WeightRule = WeightRule(conf.get('core', 'default_task_weight_rule', fallback=WeightRule.DOWNSTREAM))
DEFAULT_TRIGGER_RULE: TriggerRule = TriggerRule.ALL_SUCCESS
DEFAULT_TASK_EXECUTION_TIMEOUT: datetime.timedelta | None = conf.gettimedelta('core', 'default_task_execution_timeout')

class NotMapped(Exception):
    """Raise if a task is neither mapped nor has any parent mapped groups."""

class AbstractOperator(Templater, DAGNode):
    """Common implementation for operators, including unmapped and mapped.

    This base class is more about sharing implementations, not defining a common
    interface. Unfortunately it's difficult to use this as the common base class
    for typing due to BaseOperator carrying too much historical baggage.

    The union type ``from airflow.models.operator import Operator`` is easier
    to use for typing purposes.

    :meta private:
    """
    operator_class: type[BaseOperator] | dict[str, Any]
    weight_rule: str
    priority_weight: int
    operator_extra_links: Collection[BaseOperatorLink]
    owner: str
    task_id: str
    outlets: list
    inlets: list
    trigger_rule: TriggerRule
    _on_failure_fail_dagrun = False
    HIDE_ATTRS_FROM_UI: ClassVar[frozenset[str]] = frozenset(('log', 'dag', 'node_id', 'task_group', 'inherits_from_empty_operator', 'roots', 'leaves', 'upstream_list', 'downstream_list', 'global_operator_extra_link_dict', 'operator_extra_link_dict'))

    def get_dag(self) -> DAG | None:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError()

    @property
    def task_type(self) -> str:
        if False:
            print('Hello World!')
        raise NotImplementedError()

    @property
    def operator_name(self) -> str:
        if False:
            print('Hello World!')
        raise NotImplementedError()

    @property
    def inherits_from_empty_operator(self) -> bool:
        if False:
            return 10
        raise NotImplementedError()

    @property
    def dag_id(self) -> str:
        if False:
            return 10
        'Returns dag id if it has one or an adhoc + owner.'
        dag = self.get_dag()
        if dag:
            return dag.dag_id
        return f'adhoc_{self.owner}'

    @property
    def node_id(self) -> str:
        if False:
            return 10
        return self.task_id

    @property
    def is_setup(self) -> bool:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError()

    @is_setup.setter
    def is_setup(self, value: bool) -> None:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError()

    @property
    def is_teardown(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError()

    @is_teardown.setter
    def is_teardown(self, value: bool) -> None:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError()

    @property
    def on_failure_fail_dagrun(self):
        if False:
            i = 10
            return i + 15
        '\n        Whether the operator should fail the dagrun on failure.\n\n        :meta private:\n        '
        return self._on_failure_fail_dagrun

    @on_failure_fail_dagrun.setter
    def on_failure_fail_dagrun(self, value):
        if False:
            for i in range(10):
                print('nop')
        '\n        Setter for on_failure_fail_dagrun property.\n\n        :meta private:\n        '
        if value is True and self.is_teardown is not True:
            raise ValueError(f"Cannot set task on_failure_fail_dagrun for '{self.task_id}' because it is not a teardown task.")
        self._on_failure_fail_dagrun = value

    def as_setup(self):
        if False:
            for i in range(10):
                print('nop')
        self.is_setup = True
        return self

    def as_teardown(self, *, setups: BaseOperator | Iterable[BaseOperator] | ArgNotSet=NOTSET, on_failure_fail_dagrun=NOTSET):
        if False:
            i = 10
            return i + 15
        self.is_teardown = True
        self.trigger_rule = TriggerRule.ALL_DONE_SETUP_SUCCESS
        if on_failure_fail_dagrun is not NOTSET:
            self.on_failure_fail_dagrun = on_failure_fail_dagrun
        if not isinstance(setups, ArgNotSet):
            setups = [setups] if isinstance(setups, DependencyMixin) else setups
            for s in setups:
                s.is_setup = True
                s >> self
        return self

    def get_direct_relative_ids(self, upstream: bool=False) -> set[str]:
        if False:
            return 10
        'Get direct relative IDs to the current task, upstream or downstream.'
        if upstream:
            return self.upstream_task_ids
        return self.downstream_task_ids

    def get_flat_relative_ids(self, *, upstream: bool=False) -> set[str]:
        if False:
            i = 10
            return i + 15
        'Get a flat set of relative IDs, upstream or downstream.\n\n        Will recurse each relative found in the direction specified.\n\n        :param upstream: Whether to look for upstream or downstream relatives.\n        '
        dag = self.get_dag()
        if not dag:
            return set()
        relatives: set[str] = set()
        task_ids_to_trace = self.get_direct_relative_ids(upstream)
        while task_ids_to_trace:
            task_ids_to_trace_next: set[str] = set()
            for task_id in task_ids_to_trace:
                if task_id in relatives:
                    continue
                task_ids_to_trace_next.update(dag.task_dict[task_id].get_direct_relative_ids(upstream))
                relatives.add(task_id)
            task_ids_to_trace = task_ids_to_trace_next
        return relatives

    def get_flat_relatives(self, upstream: bool=False) -> Collection[Operator]:
        if False:
            print('Hello World!')
        'Get a flat list of relatives, either upstream or downstream.'
        dag = self.get_dag()
        if not dag:
            return set()
        return [dag.task_dict[task_id] for task_id in self.get_flat_relative_ids(upstream=upstream)]

    def get_upstreams_follow_setups(self) -> Iterable[Operator]:
        if False:
            i = 10
            return i + 15
        'All upstreams and, for each upstream setup, its respective teardowns.'
        for task in self.get_flat_relatives(upstream=True):
            yield task
            if task.is_setup:
                for t in task.downstream_list:
                    if t.is_teardown and t != self:
                        yield t

    def get_upstreams_only_setups_and_teardowns(self) -> Iterable[Operator]:
        if False:
            print('Hello World!')
        '\n        Only *relevant* upstream setups and their teardowns.\n\n        This method is meant to be used when we are clearing the task (non-upstream) and we need\n        to add in the *relevant* setups and their teardowns.\n\n        Relevant in this case means, the setup has a teardown that is downstream of ``self``,\n        or the setup has no teardowns.\n        '
        downstream_teardown_ids = {x.task_id for x in self.get_flat_relatives(upstream=False) if x.is_teardown}
        for task in self.get_flat_relatives(upstream=True):
            if not task.is_setup:
                continue
            has_no_teardowns = not any((True for x in task.downstream_list if x.is_teardown))
            if has_no_teardowns or task.downstream_task_ids.intersection(downstream_teardown_ids):
                yield task
                for t in task.downstream_list:
                    if t.is_teardown and t != self:
                        yield t

    def get_upstreams_only_setups(self) -> Iterable[Operator]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Return relevant upstream setups.\n\n        This method is meant to be used when we are checking task dependencies where we need\n        to wait for all the upstream setups to complete before we can run the task.\n        '
        for task in self.get_upstreams_only_setups_and_teardowns():
            if task.is_setup:
                yield task

    def _iter_all_mapped_downstreams(self) -> Iterator[MappedOperator | MappedTaskGroup]:
        if False:
            return 10
        "Return mapped nodes that are direct dependencies of the current task.\n\n        For now, this walks the entire DAG to find mapped nodes that has this\n        current task as an upstream. We cannot use ``downstream_list`` since it\n        only contains operators, not task groups. In the future, we should\n        provide a way to record an DAG node's all downstream nodes instead.\n\n        Note that this does not guarantee the returned tasks actually use the\n        current task for task mapping, but only checks those task are mapped\n        operators, and are downstreams of the current task.\n\n        To get a list of tasks that uses the current task for task mapping, use\n        :meth:`iter_mapped_dependants` instead.\n        "
        from airflow.models.mappedoperator import MappedOperator
        from airflow.utils.task_group import TaskGroup

        def _walk_group(group: TaskGroup) -> Iterable[tuple[str, DAGNode]]:
            if False:
                return 10
            'Recursively walk children in a task group.\n\n            This yields all direct children (including both tasks and task\n            groups), and all children of any task groups.\n            '
            for (key, child) in group.children.items():
                yield (key, child)
                if isinstance(child, TaskGroup):
                    yield from _walk_group(child)
        dag = self.get_dag()
        if not dag:
            raise RuntimeError('Cannot check for mapped dependants when not attached to a DAG')
        for (key, child) in _walk_group(dag.task_group):
            if key == self.node_id:
                continue
            if not isinstance(child, (MappedOperator, MappedTaskGroup)):
                continue
            if self.node_id in child.upstream_task_ids:
                yield child

    def iter_mapped_dependants(self) -> Iterator[MappedOperator | MappedTaskGroup]:
        if False:
            while True:
                i = 10
        "Return mapped nodes that depend on the current task the expansion.\n\n        For now, this walks the entire DAG to find mapped nodes that has this\n        current task as an upstream. We cannot use ``downstream_list`` since it\n        only contains operators, not task groups. In the future, we should\n        provide a way to record an DAG node's all downstream nodes instead.\n        "
        return (downstream for downstream in self._iter_all_mapped_downstreams() if any((p.node_id == self.node_id for p in downstream.iter_mapped_dependencies())))

    def iter_mapped_task_groups(self) -> Iterator[MappedTaskGroup]:
        if False:
            for i in range(10):
                print('nop')
        'Return mapped task groups this task belongs to.\n\n        Groups are returned from the innermost to the outmost.\n\n        :meta private:\n        '
        if (group := self.task_group) is None:
            return
        yield from group.iter_mapped_task_groups()

    def get_closest_mapped_task_group(self) -> MappedTaskGroup | None:
        if False:
            for i in range(10):
                print('nop')
        'Get the mapped task group "closest" to this task in the DAG.\n\n        :meta private:\n        '
        return next(self.iter_mapped_task_groups(), None)

    def unmap(self, resolve: None | dict[str, Any] | tuple[Context, Session]) -> BaseOperator:
        if False:
            return 10
        'Get the "normal" operator from current abstract operator.\n\n        MappedOperator uses this to unmap itself based on the map index. A non-\n        mapped operator (i.e. BaseOperator subclass) simply returns itself.\n\n        :meta private:\n        '
        raise NotImplementedError()

    @property
    def priority_weight_total(self) -> int:
        if False:
            return 10
        '\n        Total priority weight for the task. It might include all upstream or downstream tasks.\n\n        Depending on the weight rule:\n\n        - WeightRule.ABSOLUTE - only own weight\n        - WeightRule.DOWNSTREAM - adds priority weight of all downstream tasks\n        - WeightRule.UPSTREAM - adds priority weight of all upstream tasks\n        '
        if self.weight_rule == WeightRule.ABSOLUTE:
            return self.priority_weight
        elif self.weight_rule == WeightRule.DOWNSTREAM:
            upstream = False
        elif self.weight_rule == WeightRule.UPSTREAM:
            upstream = True
        else:
            upstream = False
        dag = self.get_dag()
        if dag is None:
            return self.priority_weight
        return self.priority_weight + sum((dag.task_dict[task_id].priority_weight for task_id in self.get_flat_relative_ids(upstream=upstream)))

    @cached_property
    def operator_extra_link_dict(self) -> dict[str, Any]:
        if False:
            print('Hello World!')
        'Returns dictionary of all extra links for the operator.'
        op_extra_links_from_plugin: dict[str, Any] = {}
        from airflow import plugins_manager
        plugins_manager.initialize_extra_operators_links_plugins()
        if plugins_manager.operator_extra_links is None:
            raise AirflowException("Can't load operators")
        for ope in plugins_manager.operator_extra_links:
            if ope.operators and self.operator_class in ope.operators:
                op_extra_links_from_plugin.update({ope.name: ope})
        operator_extra_links_all = {link.name: link for link in self.operator_extra_links}
        operator_extra_links_all.update(op_extra_links_from_plugin)
        return operator_extra_links_all

    @cached_property
    def global_operator_extra_link_dict(self) -> dict[str, Any]:
        if False:
            return 10
        'Returns dictionary of all global extra links.'
        from airflow import plugins_manager
        plugins_manager.initialize_extra_operators_links_plugins()
        if plugins_manager.global_operator_extra_links is None:
            raise AirflowException("Can't load operators")
        return {link.name: link for link in plugins_manager.global_operator_extra_links}

    @cached_property
    def extra_links(self) -> list[str]:
        if False:
            i = 10
            return i + 15
        return sorted(set(self.operator_extra_link_dict).union(self.global_operator_extra_link_dict))

    def get_extra_links(self, ti: TaskInstance, link_name: str) -> str | None:
        if False:
            return 10
        "For an operator, gets the URLs that the ``extra_links`` entry points to.\n\n        :meta private:\n\n        :raise ValueError: The error message of a ValueError will be passed on through to\n            the fronted to show up as a tooltip on the disabled link.\n        :param ti: The TaskInstance for the URL being searched for.\n        :param link_name: The name of the link we're looking for the URL for. Should be\n            one of the options specified in ``extra_links``.\n        "
        link: BaseOperatorLink | None = self.operator_extra_link_dict.get(link_name)
        if not link:
            link = self.global_operator_extra_link_dict.get(link_name)
            if not link:
                return None
        parameters = inspect.signature(link.get_link).parameters
        old_signature = all((name != 'ti_key' for (name, p) in parameters.items() if p.kind != p.VAR_KEYWORD))
        if old_signature:
            return link.get_link(self.unmap(None), ti.dag_run.logical_date)
        return link.get_link(self.unmap(None), ti_key=ti.key)

    @cache
    def get_parse_time_mapped_ti_count(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the number of mapped task instances that can be created on DAG run creation.\n\n        This only considers literal mapped arguments, and would return *None*\n        when any non-literal values are used for mapping.\n\n        :raise NotFullyPopulated: If non-literal mapped arguments are encountered.\n        :raise NotMapped: If the operator is neither mapped, nor has any parent\n            mapped task groups.\n        :return: Total number of mapped TIs this task should have.\n        '
        group = self.get_closest_mapped_task_group()
        if group is None:
            raise NotMapped
        return group.get_parse_time_mapped_ti_count()

    def get_mapped_ti_count(self, run_id: str, *, session: Session) -> int:
        if False:
            return 10
        '\n        Return the number of mapped TaskInstances that can be created at run time.\n\n        This considers both literal and non-literal mapped arguments, and the\n        result is therefore available when all depended tasks have finished. The\n        return value should be identical to ``parse_time_mapped_ti_count`` if\n        all mapped arguments are literal.\n\n        :raise NotFullyPopulated: If upstream tasks are not all complete yet.\n        :raise NotMapped: If the operator is neither mapped, nor has any parent\n            mapped task groups.\n        :return: Total number of mapped TIs this task should have.\n        '
        group = self.get_closest_mapped_task_group()
        if group is None:
            raise NotMapped
        return group.get_mapped_ti_count(run_id, session=session)

    def expand_mapped_task(self, run_id: str, *, session: Session) -> tuple[Sequence[TaskInstance], int]:
        if False:
            i = 10
            return i + 15
        'Create the mapped task instances for mapped task.\n\n        :raise NotMapped: If this task does not need expansion.\n        :return: The newly created mapped task instances (if any) in ascending\n            order by map index, and the maximum map index value.\n        '
        from sqlalchemy import func, or_
        from airflow.models.baseoperator import BaseOperator
        from airflow.models.mappedoperator import MappedOperator
        from airflow.models.taskinstance import TaskInstance
        from airflow.settings import task_instance_mutation_hook
        if not isinstance(self, (BaseOperator, MappedOperator)):
            raise RuntimeError(f'cannot expand unrecognized operator type {type(self).__name__}')
        try:
            total_length: int | None = self.get_mapped_ti_count(run_id, session=session)
        except NotFullyPopulated as e:
            if not self.dag or not self.dag.partial:
                self.log.error('Cannot expand %r for run %s; missing upstream values: %s', self, run_id, sorted(e.missing))
            total_length = None
        state: TaskInstanceState | None = None
        unmapped_ti: TaskInstance | None = session.scalars(select(TaskInstance).where(TaskInstance.dag_id == self.dag_id, TaskInstance.task_id == self.task_id, TaskInstance.run_id == run_id, TaskInstance.map_index == -1, or_(TaskInstance.state.in_(State.unfinished), TaskInstance.state.is_(None)))).one_or_none()
        all_expanded_tis: list[TaskInstance] = []
        if unmapped_ti:
            if total_length is None:
                if not self.dag or not self.dag.partial:
                    unmapped_ti.state = TaskInstanceState.UPSTREAM_FAILED
            elif total_length < 1:
                self.log.info('Marking %s as SKIPPED since the map has %d values to expand', unmapped_ti, total_length)
                unmapped_ti.state = TaskInstanceState.SKIPPED
            else:
                zero_index_ti_exists = exists_query(TaskInstance.dag_id == self.dag_id, TaskInstance.task_id == self.task_id, TaskInstance.run_id == run_id, TaskInstance.map_index == 0, session=session)
                if not zero_index_ti_exists:
                    unmapped_ti.map_index = 0
                    self.log.debug('Updated in place to become %s', unmapped_ti)
                    all_expanded_tis.append(unmapped_ti)
                    session.flush()
                else:
                    self.log.debug('Deleting the original task instance: %s', unmapped_ti)
                    session.delete(unmapped_ti)
                state = unmapped_ti.state
        if total_length is None or total_length < 1:
            indexes_to_map: Iterable[int] = ()
        else:
            current_max_mapping = session.scalar(select(func.max(TaskInstance.map_index)).where(TaskInstance.dag_id == self.dag_id, TaskInstance.task_id == self.task_id, TaskInstance.run_id == run_id))
            indexes_to_map = range(current_max_mapping + 1, total_length)
        for index in indexes_to_map:
            ti = TaskInstance(self, run_id=run_id, map_index=index, state=state)
            self.log.debug('Expanding TIs upserted %s', ti)
            task_instance_mutation_hook(ti)
            ti = session.merge(ti)
            ti.refresh_from_task(self)
            all_expanded_tis.append(ti)
        total_expanded_ti_count = total_length or 0
        query = select(TaskInstance).where(TaskInstance.dag_id == self.dag_id, TaskInstance.task_id == self.task_id, TaskInstance.run_id == run_id, TaskInstance.map_index >= total_expanded_ti_count)
        query = with_row_locks(query, of=TaskInstance, session=session, **skip_locked(session=session))
        to_update = session.scalars(query)
        for ti in to_update:
            ti.state = TaskInstanceState.REMOVED
        session.flush()
        return (all_expanded_tis, total_expanded_ti_count - 1)

    def render_template_fields(self, context: Context, jinja_env: jinja2.Environment | None=None) -> None:
        if False:
            return 10
        'Template all attributes listed in *self.template_fields*.\n\n        If the operator is mapped, this should return the unmapped, fully\n        rendered, and map-expanded operator. The mapped operator should not be\n        modified. However, *context* may be modified in-place to reference the\n        unmapped operator for template rendering.\n\n        If the operator is not mapped, this should modify the operator in-place.\n        '
        raise NotImplementedError()

    def _render(self, template, context, dag: DAG | None=None):
        if False:
            for i in range(10):
                print('nop')
        if dag is None:
            dag = self.get_dag()
        return super()._render(template, context, dag=dag)

    def get_template_env(self, dag: DAG | None=None) -> jinja2.Environment:
        if False:
            i = 10
            return i + 15
        'Get the template environment for rendering templates.'
        if dag is None:
            dag = self.get_dag()
        return super().get_template_env(dag=dag)

    @provide_session
    def _do_render_template_fields(self, parent: Any, template_fields: Iterable[str], context: Context, jinja_env: jinja2.Environment, seen_oids: set[int], *, session: Session=NEW_SESSION) -> None:
        if False:
            i = 10
            return i + 15
        'Override the base to use custom error logging.'
        for attr_name in template_fields:
            try:
                value = getattr(parent, attr_name)
            except AttributeError:
                raise AttributeError(f'{attr_name!r} is configured as a template field but {parent.task_type} does not have this attribute.')
            try:
                if not value:
                    continue
            except Exception:
                self.log.info("Unable to check if the value of type '%s' is False for task '%s', field '%s'.", type(value).__name__, self.task_id, attr_name)
                pass
            try:
                rendered_content = self.render_template(value, context, jinja_env, seen_oids)
            except Exception:
                value_masked = redact(name=attr_name, value=value)
                self.log.exception("Exception rendering Jinja template for task '%s', field '%s'. Template: %r", self.task_id, attr_name, value_masked)
                raise
            else:
                setattr(parent, attr_name, rendered_content)

    def __enter__(self):
        if False:
            print('Hello World!')
        if not self.is_setup and (not self.is_teardown):
            raise AirflowException('Only setup/teardown tasks can be used as context managers.')
        SetupTeardownContext.push_setup_teardown_task(self)
        return SetupTeardownContext

    def __exit__(self, exc_type, exc_val, exc_tb):
        if False:
            for i in range(10):
                print('nop')
        SetupTeardownContext.set_work_task_roots_and_leaves()