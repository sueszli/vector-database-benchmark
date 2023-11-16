from __future__ import annotations
import collections.abc
import contextlib
import copy
import warnings
from typing import TYPE_CHECKING, Any, ClassVar, Collection, Iterable, Iterator, Mapping, Sequence, Union
import attr
from airflow.compat.functools import cache
from airflow.exceptions import AirflowException, UnmappableOperator
from airflow.models.abstractoperator import DEFAULT_IGNORE_FIRST_DEPENDS_ON_PAST, DEFAULT_OWNER, DEFAULT_POOL_SLOTS, DEFAULT_PRIORITY_WEIGHT, DEFAULT_QUEUE, DEFAULT_RETRIES, DEFAULT_RETRY_DELAY, DEFAULT_TRIGGER_RULE, DEFAULT_WAIT_FOR_PAST_DEPENDS_BEFORE_SKIPPING, DEFAULT_WEIGHT_RULE, AbstractOperator, NotMapped
from airflow.models.expandinput import DictOfListsExpandInput, ListOfDictsExpandInput, is_mappable
from airflow.models.pool import Pool
from airflow.serialization.enums import DagAttributeTypes
from airflow.ti_deps.deps.mapped_task_expanded import MappedTaskIsExpanded
from airflow.typing_compat import Literal
from airflow.utils.context import context_update_for_unmapped
from airflow.utils.helpers import is_container, prevent_duplicates
from airflow.utils.task_instance_session import get_current_task_instance_session
from airflow.utils.types import NOTSET
from airflow.utils.xcom import XCOM_RETURN_KEY
if TYPE_CHECKING:
    import datetime
    import jinja2
    import pendulum
    from sqlalchemy.orm.session import Session
    from airflow.models.abstractoperator import TaskStateChangeCallback
    from airflow.models.baseoperator import BaseOperator, BaseOperatorLink
    from airflow.models.dag import DAG
    from airflow.models.expandinput import ExpandInput, OperatorExpandArgument, OperatorExpandKwargsArgument
    from airflow.models.operator import Operator
    from airflow.models.param import ParamsDict
    from airflow.models.xcom_arg import XComArg
    from airflow.ti_deps.deps.base_ti_dep import BaseTIDep
    from airflow.utils.context import Context
    from airflow.utils.operator_resources import Resources
    from airflow.utils.task_group import TaskGroup
    from airflow.utils.trigger_rule import TriggerRule
ValidationSource = Union[Literal['expand'], Literal['partial']]

def validate_mapping_kwargs(op: type[BaseOperator], func: ValidationSource, value: dict[str, Any]) -> None:
    if False:
        i = 10
        return i + 15
    unknown_args = value.copy()
    for klass in op.mro():
        init = klass.__init__
        try:
            param_names = init._BaseOperatorMeta__param_names
        except AttributeError:
            continue
        for name in param_names:
            value = unknown_args.pop(name, NOTSET)
            if func != 'expand':
                continue
            if value is NOTSET:
                continue
            if is_mappable(value):
                continue
            type_name = type(value).__name__
            error = f'{op.__name__}.expand() got an unexpected type {type_name!r} for keyword argument {name}'
            raise ValueError(error)
        if not unknown_args:
            return
    if len(unknown_args) == 1:
        error = f'an unexpected keyword argument {unknown_args.popitem()[0]!r}'
    else:
        names = ', '.join((repr(n) for n in unknown_args))
        error = f'unexpected keyword arguments {names}'
    raise TypeError(f'{op.__name__}.{func}() got {error}')

def ensure_xcomarg_return_value(arg: Any) -> None:
    if False:
        return 10
    from airflow.models.xcom_arg import XComArg
    if isinstance(arg, XComArg):
        for (operator, key) in arg.iter_references():
            if key != XCOM_RETURN_KEY:
                raise ValueError(f'cannot map over XCom with custom key {key!r} from {operator}')
    elif not is_container(arg):
        return
    elif isinstance(arg, collections.abc.Mapping):
        for v in arg.values():
            ensure_xcomarg_return_value(v)
    elif isinstance(arg, collections.abc.Iterable):
        for v in arg:
            ensure_xcomarg_return_value(v)

@attr.define(kw_only=True, repr=False)
class OperatorPartial:
    """An "intermediate state" returned by ``BaseOperator.partial()``.

    This only exists at DAG-parsing time; the only intended usage is for the
    user to call ``.expand()`` on it at some point (usually in a method chain) to
    create a ``MappedOperator`` to add into the DAG.
    """
    operator_class: type[BaseOperator]
    kwargs: dict[str, Any]
    params: ParamsDict | dict
    _expand_called: bool = False

    def __attrs_post_init__(self):
        if False:
            while True:
                i = 10
        from airflow.operators.subdag import SubDagOperator
        if issubclass(self.operator_class, SubDagOperator):
            raise TypeError('Mapping over deprecated SubDagOperator is not supported')
        validate_mapping_kwargs(self.operator_class, 'partial', self.kwargs)

    def __repr__(self) -> str:
        if False:
            print('Hello World!')
        args = ', '.join((f'{k}={v!r}' for (k, v) in self.kwargs.items()))
        return f'{self.operator_class.__name__}.partial({args})'

    def __del__(self):
        if False:
            i = 10
            return i + 15
        if not self._expand_called:
            try:
                task_id = repr(self.kwargs['task_id'])
            except KeyError:
                task_id = f'at {hex(id(self))}'
            warnings.warn(f'Task {task_id} was never mapped!')

    def expand(self, **mapped_kwargs: OperatorExpandArgument) -> MappedOperator:
        if False:
            print('Hello World!')
        if not mapped_kwargs:
            raise TypeError('no arguments to expand against')
        validate_mapping_kwargs(self.operator_class, 'expand', mapped_kwargs)
        prevent_duplicates(self.kwargs, mapped_kwargs, fail_reason='unmappable or already specified')
        return self._expand(DictOfListsExpandInput(mapped_kwargs), strict=False)

    def expand_kwargs(self, kwargs: OperatorExpandKwargsArgument, *, strict: bool=True) -> MappedOperator:
        if False:
            return 10
        from airflow.models.xcom_arg import XComArg
        if isinstance(kwargs, collections.abc.Sequence):
            for item in kwargs:
                if not isinstance(item, (XComArg, collections.abc.Mapping)):
                    raise TypeError(f'expected XComArg or list[dict], not {type(kwargs).__name__}')
        elif not isinstance(kwargs, XComArg):
            raise TypeError(f'expected XComArg or list[dict], not {type(kwargs).__name__}')
        return self._expand(ListOfDictsExpandInput(kwargs), strict=strict)

    def _expand(self, expand_input: ExpandInput, *, strict: bool) -> MappedOperator:
        if False:
            i = 10
            return i + 15
        from airflow.operators.empty import EmptyOperator
        self._expand_called = True
        ensure_xcomarg_return_value(expand_input.value)
        partial_kwargs = self.kwargs.copy()
        task_id = partial_kwargs.pop('task_id')
        dag = partial_kwargs.pop('dag')
        task_group = partial_kwargs.pop('task_group')
        start_date = partial_kwargs.pop('start_date')
        end_date = partial_kwargs.pop('end_date')
        try:
            operator_name = self.operator_class.custom_operator_name
        except AttributeError:
            operator_name = self.operator_class.__name__
        op = MappedOperator(operator_class=self.operator_class, expand_input=expand_input, partial_kwargs=partial_kwargs, task_id=task_id, params=self.params, deps=MappedOperator.deps_for(self.operator_class), operator_extra_links=self.operator_class.operator_extra_links, template_ext=self.operator_class.template_ext, template_fields=self.operator_class.template_fields, template_fields_renderers=self.operator_class.template_fields_renderers, ui_color=self.operator_class.ui_color, ui_fgcolor=self.operator_class.ui_fgcolor, is_empty=issubclass(self.operator_class, EmptyOperator), task_module=self.operator_class.__module__, task_type=self.operator_class.__name__, operator_name=operator_name, dag=dag, task_group=task_group, start_date=start_date, end_date=end_date, disallow_kwargs_override=strict, expand_input_attr='expand_input')
        return op

@attr.define(kw_only=True, getstate_setstate=False)
class MappedOperator(AbstractOperator):
    """Object representing a mapped operator in a DAG."""
    operator_class: type[BaseOperator] | dict[str, Any]
    expand_input: ExpandInput
    partial_kwargs: dict[str, Any]
    task_id: str
    params: ParamsDict | dict
    deps: frozenset[BaseTIDep]
    operator_extra_links: Collection[BaseOperatorLink]
    template_ext: Sequence[str]
    template_fields: Collection[str]
    template_fields_renderers: dict[str, str]
    ui_color: str
    ui_fgcolor: str
    _is_empty: bool
    _task_module: str
    _task_type: str
    _operator_name: str
    dag: DAG | None
    task_group: TaskGroup | None
    start_date: pendulum.DateTime | None
    end_date: pendulum.DateTime | None
    upstream_task_ids: set[str] = attr.ib(factory=set, init=False)
    downstream_task_ids: set[str] = attr.ib(factory=set, init=False)
    _disallow_kwargs_override: bool
    'Whether execution fails if ``expand_input`` has duplicates to ``partial_kwargs``.\n\n    If *False*, values from ``expand_input`` under duplicate keys override those\n    under corresponding keys in ``partial_kwargs``.\n    '
    _expand_input_attr: str
    'Where to get kwargs to calculate expansion length against.\n\n    This should be a name to call ``getattr()`` on.\n    '
    subdag: None = None
    supports_lineage: bool = False
    HIDE_ATTRS_FROM_UI: ClassVar[frozenset[str]] = AbstractOperator.HIDE_ATTRS_FROM_UI | frozenset(('parse_time_mapped_ti_count', 'operator_class'))

    def __hash__(self):
        if False:
            i = 10
            return i + 15
        return id(self)

    def __repr__(self):
        if False:
            return 10
        return f'<Mapped({self._task_type}): {self.task_id}>'

    def __attrs_post_init__(self):
        if False:
            for i in range(10):
                print('nop')
        from airflow.models.xcom_arg import XComArg
        if self.get_closest_mapped_task_group() is not None:
            raise NotImplementedError('operator expansion in an expanded task group is not yet supported')
        if self.task_group:
            self.task_group.add(self)
        if self.dag:
            self.dag.add_task(self)
        XComArg.apply_upstream_relationship(self, self.expand_input.value)
        for (k, v) in self.partial_kwargs.items():
            if k in self.template_fields:
                XComArg.apply_upstream_relationship(self, v)
        if self.partial_kwargs.get('sla') is not None:
            raise AirflowException(f'SLAs are unsupported with mapped tasks. Please set `sla=None` for task {self.task_id!r}.')

    @classmethod
    @cache
    def get_serialized_fields(cls):
        if False:
            for i in range(10):
                print('nop')
        return frozenset(attr.fields_dict(MappedOperator)) - {'dag', 'deps', 'expand_input', 'subdag', 'task_group', 'upstream_task_ids', 'supports_lineage', '_is_setup', '_is_teardown', '_on_failure_fail_dagrun'}

    @staticmethod
    @cache
    def deps_for(operator_class: type[BaseOperator]) -> frozenset[BaseTIDep]:
        if False:
            while True:
                i = 10
        operator_deps = operator_class.deps
        if not isinstance(operator_deps, collections.abc.Set):
            raise UnmappableOperator(f"'deps' must be a set defined as a class-level variable on {operator_class.__name__}, not a {type(operator_deps).__name__}")
        return operator_deps | {MappedTaskIsExpanded()}

    @property
    def task_type(self) -> str:
        if False:
            while True:
                i = 10
        'Implementing Operator.'
        return self._task_type

    @property
    def operator_name(self) -> str:
        if False:
            print('Hello World!')
        return self._operator_name

    @property
    def inherits_from_empty_operator(self) -> bool:
        if False:
            return 10
        'Implementing Operator.'
        return self._is_empty

    @property
    def roots(self) -> Sequence[AbstractOperator]:
        if False:
            print('Hello World!')
        'Implementing DAGNode.'
        return [self]

    @property
    def leaves(self) -> Sequence[AbstractOperator]:
        if False:
            for i in range(10):
                print('nop')
        'Implementing DAGNode.'
        return [self]

    @property
    def owner(self) -> str:
        if False:
            return 10
        return self.partial_kwargs.get('owner', DEFAULT_OWNER)

    @property
    def email(self) -> None | str | Iterable[str]:
        if False:
            while True:
                i = 10
        return self.partial_kwargs.get('email')

    @property
    def trigger_rule(self) -> TriggerRule:
        if False:
            i = 10
            return i + 15
        return self.partial_kwargs.get('trigger_rule', DEFAULT_TRIGGER_RULE)

    @trigger_rule.setter
    def trigger_rule(self, value):
        if False:
            while True:
                i = 10
        self.partial_kwargs['trigger_rule'] = value

    @property
    def is_setup(self) -> bool:
        if False:
            print('Hello World!')
        return bool(self.partial_kwargs.get('is_setup'))

    @is_setup.setter
    def is_setup(self, value: bool) -> None:
        if False:
            while True:
                i = 10
        self.partial_kwargs['is_setup'] = value

    @property
    def is_teardown(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return bool(self.partial_kwargs.get('is_teardown'))

    @is_teardown.setter
    def is_teardown(self, value: bool) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.partial_kwargs['is_teardown'] = value

    @property
    def depends_on_past(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return bool(self.partial_kwargs.get('depends_on_past'))

    @property
    def ignore_first_depends_on_past(self) -> bool:
        if False:
            i = 10
            return i + 15
        value = self.partial_kwargs.get('ignore_first_depends_on_past', DEFAULT_IGNORE_FIRST_DEPENDS_ON_PAST)
        return bool(value)

    @property
    def wait_for_past_depends_before_skipping(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        value = self.partial_kwargs.get('wait_for_past_depends_before_skipping', DEFAULT_WAIT_FOR_PAST_DEPENDS_BEFORE_SKIPPING)
        return bool(value)

    @property
    def wait_for_downstream(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return bool(self.partial_kwargs.get('wait_for_downstream'))

    @property
    def retries(self) -> int | None:
        if False:
            while True:
                i = 10
        return self.partial_kwargs.get('retries', DEFAULT_RETRIES)

    @property
    def queue(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return self.partial_kwargs.get('queue', DEFAULT_QUEUE)

    @property
    def pool(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return self.partial_kwargs.get('pool', Pool.DEFAULT_POOL_NAME)

    @property
    def pool_slots(self) -> str | None:
        if False:
            for i in range(10):
                print('nop')
        return self.partial_kwargs.get('pool_slots', DEFAULT_POOL_SLOTS)

    @property
    def execution_timeout(self) -> datetime.timedelta | None:
        if False:
            return 10
        return self.partial_kwargs.get('execution_timeout')

    @property
    def max_retry_delay(self) -> datetime.timedelta | None:
        if False:
            while True:
                i = 10
        return self.partial_kwargs.get('max_retry_delay')

    @property
    def retry_delay(self) -> datetime.timedelta:
        if False:
            while True:
                i = 10
        return self.partial_kwargs.get('retry_delay', DEFAULT_RETRY_DELAY)

    @property
    def retry_exponential_backoff(self) -> bool:
        if False:
            return 10
        return bool(self.partial_kwargs.get('retry_exponential_backoff'))

    @property
    def priority_weight(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return self.partial_kwargs.get('priority_weight', DEFAULT_PRIORITY_WEIGHT)

    @property
    def weight_rule(self) -> str:
        if False:
            print('Hello World!')
        return self.partial_kwargs.get('weight_rule', DEFAULT_WEIGHT_RULE)

    @property
    def sla(self) -> datetime.timedelta | None:
        if False:
            print('Hello World!')
        return self.partial_kwargs.get('sla')

    @property
    def max_active_tis_per_dag(self) -> int | None:
        if False:
            i = 10
            return i + 15
        return self.partial_kwargs.get('max_active_tis_per_dag')

    @property
    def max_active_tis_per_dagrun(self) -> int | None:
        if False:
            for i in range(10):
                print('nop')
        return self.partial_kwargs.get('max_active_tis_per_dagrun')

    @property
    def resources(self) -> Resources | None:
        if False:
            while True:
                i = 10
        return self.partial_kwargs.get('resources')

    @property
    def on_execute_callback(self) -> None | TaskStateChangeCallback | list[TaskStateChangeCallback]:
        if False:
            print('Hello World!')
        return self.partial_kwargs.get('on_execute_callback')

    @on_execute_callback.setter
    def on_execute_callback(self, value: TaskStateChangeCallback | None) -> None:
        if False:
            i = 10
            return i + 15
        self.partial_kwargs['on_execute_callback'] = value

    @property
    def on_failure_callback(self) -> None | TaskStateChangeCallback | list[TaskStateChangeCallback]:
        if False:
            i = 10
            return i + 15
        return self.partial_kwargs.get('on_failure_callback')

    @on_failure_callback.setter
    def on_failure_callback(self, value: TaskStateChangeCallback | None) -> None:
        if False:
            i = 10
            return i + 15
        self.partial_kwargs['on_failure_callback'] = value

    @property
    def on_retry_callback(self) -> None | TaskStateChangeCallback | list[TaskStateChangeCallback]:
        if False:
            i = 10
            return i + 15
        return self.partial_kwargs.get('on_retry_callback')

    @on_retry_callback.setter
    def on_retry_callback(self, value: TaskStateChangeCallback | None) -> None:
        if False:
            while True:
                i = 10
        self.partial_kwargs['on_retry_callback'] = value

    @property
    def on_success_callback(self) -> None | TaskStateChangeCallback | list[TaskStateChangeCallback]:
        if False:
            return 10
        return self.partial_kwargs.get('on_success_callback')

    @on_success_callback.setter
    def on_success_callback(self, value: TaskStateChangeCallback | None) -> None:
        if False:
            i = 10
            return i + 15
        self.partial_kwargs['on_success_callback'] = value

    @property
    def run_as_user(self) -> str | None:
        if False:
            print('Hello World!')
        return self.partial_kwargs.get('run_as_user')

    @property
    def executor_config(self) -> dict:
        if False:
            i = 10
            return i + 15
        return self.partial_kwargs.get('executor_config', {})

    @property
    def inlets(self) -> list[Any]:
        if False:
            return 10
        return self.partial_kwargs.get('inlets', [])

    @inlets.setter
    def inlets(self, value: list[Any]) -> None:
        if False:
            while True:
                i = 10
        self.partial_kwargs['inlets'] = value

    @property
    def outlets(self) -> list[Any]:
        if False:
            return 10
        return self.partial_kwargs.get('outlets', [])

    @outlets.setter
    def outlets(self, value: list[Any]) -> None:
        if False:
            print('Hello World!')
        self.partial_kwargs['outlets'] = value

    @property
    def doc(self) -> str | None:
        if False:
            print('Hello World!')
        return self.partial_kwargs.get('doc')

    @property
    def doc_md(self) -> str | None:
        if False:
            for i in range(10):
                print('nop')
        return self.partial_kwargs.get('doc_md')

    @property
    def doc_json(self) -> str | None:
        if False:
            while True:
                i = 10
        return self.partial_kwargs.get('doc_json')

    @property
    def doc_yaml(self) -> str | None:
        if False:
            return 10
        return self.partial_kwargs.get('doc_yaml')

    @property
    def doc_rst(self) -> str | None:
        if False:
            for i in range(10):
                print('nop')
        return self.partial_kwargs.get('doc_rst')

    def get_dag(self) -> DAG | None:
        if False:
            for i in range(10):
                print('nop')
        'Implement Operator.'
        return self.dag

    @property
    def output(self) -> XComArg:
        if False:
            for i in range(10):
                print('nop')
        'Return reference to XCom pushed by current operator.'
        from airflow.models.xcom_arg import XComArg
        return XComArg(operator=self)

    def serialize_for_task_group(self) -> tuple[DagAttributeTypes, Any]:
        if False:
            return 10
        'Implement DAGNode.'
        return (DagAttributeTypes.OP, self.task_id)

    def _expand_mapped_kwargs(self, context: Context, session: Session) -> tuple[Mapping[str, Any], set[int]]:
        if False:
            while True:
                i = 10
        'Get the kwargs to create the unmapped operator.\n\n        This exists because taskflow operators expand against op_kwargs, not the\n        entire operator kwargs dict.\n        '
        return self._get_specified_expand_input().resolve(context, session)

    def _get_unmap_kwargs(self, mapped_kwargs: Mapping[str, Any], *, strict: bool) -> dict[str, Any]:
        if False:
            print('Hello World!')
        'Get init kwargs to unmap the underlying operator class.\n\n        :param mapped_kwargs: The dict returned by ``_expand_mapped_kwargs``.\n        '
        if strict:
            prevent_duplicates(self.partial_kwargs, mapped_kwargs, fail_reason='unmappable or already specified')
        params = copy.copy(self.params)
        with contextlib.suppress(KeyError):
            params.update(mapped_kwargs['params'])
        return {'task_id': self.task_id, 'dag': self.dag, 'task_group': self.task_group, 'start_date': self.start_date, 'end_date': self.end_date, **self.partial_kwargs, **mapped_kwargs, 'params': params}

    def unmap(self, resolve: None | Mapping[str, Any] | tuple[Context, Session]) -> BaseOperator:
        if False:
            print('Hello World!')
        'Get the "normal" Operator after applying the current mapping.\n\n        The *resolve* argument is only used if ``operator_class`` is a real\n        class, i.e. if this operator is not serialized. If ``operator_class`` is\n        not a class (i.e. this DAG has been deserialized), this returns a\n        SerializedBaseOperator that "looks like" the actual unmapping result.\n\n        If *resolve* is a two-tuple (context, session), the information is used\n        to resolve the mapped arguments into init arguments. If it is a mapping,\n        no resolving happens, the mapping directly provides those init arguments\n        resolved from mapped kwargs.\n\n        :meta private:\n        '
        if isinstance(self.operator_class, type):
            if isinstance(resolve, collections.abc.Mapping):
                kwargs = resolve
            elif resolve is not None:
                (kwargs, _) = self._expand_mapped_kwargs(*resolve)
            else:
                raise RuntimeError('cannot unmap a non-serialized operator without context')
            kwargs = self._get_unmap_kwargs(kwargs, strict=self._disallow_kwargs_override)
            is_setup = kwargs.pop('is_setup', False)
            is_teardown = kwargs.pop('is_teardown', False)
            on_failure_fail_dagrun = kwargs.pop('on_failure_fail_dagrun', False)
            op = self.operator_class(**kwargs, _airflow_from_mapped=True)
            op.task_id = self.task_id
            op.is_setup = is_setup
            op.is_teardown = is_teardown
            op.on_failure_fail_dagrun = on_failure_fail_dagrun
            return op
        from airflow.serialization.serialized_objects import SerializedBaseOperator
        op = SerializedBaseOperator(task_id=self.task_id, params=self.params, _airflow_from_mapped=True)
        SerializedBaseOperator.populate_operator(op, self.operator_class)
        if self.dag is not None:
            SerializedBaseOperator.set_task_dag_references(op, self.dag)
        return op

    def _get_specified_expand_input(self) -> ExpandInput:
        if False:
            return 10
        'Input received from the expand call on the operator.'
        return getattr(self, self._expand_input_attr)

    def prepare_for_execution(self) -> MappedOperator:
        if False:
            return 10
        return self

    def iter_mapped_dependencies(self) -> Iterator[Operator]:
        if False:
            i = 10
            return i + 15
        'Upstream dependencies that provide XComs used by this task for task mapping.'
        from airflow.models.xcom_arg import XComArg
        for (operator, _) in XComArg.iter_xcom_references(self._get_specified_expand_input()):
            yield operator

    @cache
    def get_parse_time_mapped_ti_count(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        current_count = self._get_specified_expand_input().get_parse_time_mapped_ti_count()
        try:
            parent_count = super().get_parse_time_mapped_ti_count()
        except NotMapped:
            return current_count
        return parent_count * current_count

    def get_mapped_ti_count(self, run_id: str, *, session: Session) -> int:
        if False:
            while True:
                i = 10
        current_count = self._get_specified_expand_input().get_total_map_length(run_id, session=session)
        try:
            parent_count = super().get_mapped_ti_count(run_id, session=session)
        except NotMapped:
            return current_count
        return parent_count * current_count

    def render_template_fields(self, context: Context, jinja_env: jinja2.Environment | None=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Template all attributes listed in *self.template_fields*.\n\n        This updates *context* to reference the map-expanded task and relevant\n        information, without modifying the mapped operator. The expanded task\n        in *context* is then rendered in-place.\n\n        :param context: Context dict with values to apply on content.\n        :param jinja_env: Jinja environment to use for rendering.\n        '
        if not jinja_env:
            jinja_env = self.get_template_env()
        session = get_current_task_instance_session()
        (mapped_kwargs, seen_oids) = self._expand_mapped_kwargs(context, session)
        unmapped_task = self.unmap(mapped_kwargs)
        context_update_for_unmapped(context, unmapped_task)
        unmapped_task._do_render_template_fields(parent=unmapped_task, template_fields=self.template_fields, context=context, jinja_env=jinja_env, seen_oids=seen_oids, session=session)