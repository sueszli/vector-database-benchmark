"""Unit tests for stringified DAGs."""
from __future__ import annotations
import copy
import importlib
import importlib.util
import json
import multiprocessing
import os
import pickle
from datetime import datetime, timedelta
from glob import glob
from pathlib import Path
from typing import TYPE_CHECKING
from unittest import mock
import attr
import pendulum
import pytest
from dateutil.relativedelta import FR, relativedelta
from kubernetes.client import models as k8s
import airflow
from airflow.datasets import Dataset
from airflow.decorators import teardown
from airflow.decorators.base import DecoratedOperator
from airflow.exceptions import AirflowException, SerializationError
from airflow.hooks.base import BaseHook
from airflow.models.baseoperator import BaseOperator, BaseOperatorLink
from airflow.models.connection import Connection
from airflow.models.dag import DAG
from airflow.models.dagbag import DagBag
from airflow.models.expandinput import EXPAND_INPUT_EMPTY
from airflow.models.mappedoperator import MappedOperator
from airflow.models.param import Param, ParamsDict
from airflow.models.xcom import XCom
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.providers.cncf.kubernetes.pod_generator import PodGenerator
from airflow.security import permissions
from airflow.sensors.bash import BashSensor
from airflow.serialization.enums import Encoding
from airflow.serialization.json_schema import load_dag_schema_dict
from airflow.serialization.serialized_objects import DagDependency, DependencyDetector, SerializedBaseOperator, SerializedDAG
from airflow.ti_deps.deps.base_ti_dep import BaseTIDep
from airflow.timetables.simple import NullTimetable, OnceTimetable
from airflow.utils import timezone
from airflow.utils.operator_resources import Resources
from airflow.utils.task_group import TaskGroup
from airflow.utils.xcom import XCOM_RETURN_KEY
from tests.test_utils.config import conf_vars
from tests.test_utils.mock_operators import AirflowLink2, CustomOperator, GoogleLink, MockOperator
from tests.test_utils.timetables import CustomSerializationTimetable, cron_timetable, delta_timetable
if TYPE_CHECKING:
    from airflow.models.operator import Operator
    from airflow.utils.context import Context
repo_root = Path(airflow.__file__).parent.parent

class CustomDepOperator(BashOperator):
    """
    Used for testing custom dependency detector.

    TODO: remove in Airflow 3.0
    """

class CustomDependencyDetector(DependencyDetector):
    """
    Prior to deprecation of custom dependency detector, the return type as DagDependency | None.
    This class verifies that custom dependency detector classes which assume that return type will still
    work until support for them is removed in 3.0.

    TODO: remove in Airflow 3.0
    """

    @staticmethod
    def detect_task_dependencies(task: Operator) -> DagDependency | None:
        if False:
            i = 10
            return i + 15
        if isinstance(task, CustomDepOperator):
            return DagDependency(source=task.dag_id, target='nothing', dependency_type='abc', dependency_id=task.task_id)
        else:
            return DependencyDetector().detect_task_dependencies(task)
executor_config_pod = k8s.V1Pod(metadata=k8s.V1ObjectMeta(name='my-name'), spec=k8s.V1PodSpec(containers=[k8s.V1Container(name='base', volume_mounts=[k8s.V1VolumeMount(name='my-vol', mount_path='/vol/')])]))
TYPE = Encoding.TYPE
VAR = Encoding.VAR
serialized_simple_dag_ground_truth = {'__version': 1, 'dag': {'default_args': {'__type': 'dict', '__var': {'depends_on_past': False, 'retries': 1, 'retry_delay': {'__type': 'timedelta', '__var': 300.0}, 'max_retry_delay': {'__type': 'timedelta', '__var': 600.0}, 'sla': {'__type': 'timedelta', '__var': 100.0}}}, 'start_date': 1564617600.0, '_task_group': {'_group_id': None, 'prefix_group_id': True, 'children': {'bash_task': ('operator', 'bash_task'), 'custom_task': ('operator', 'custom_task')}, 'tooltip': '', 'ui_color': 'CornflowerBlue', 'ui_fgcolor': '#000', 'upstream_group_ids': [], 'downstream_group_ids': [], 'upstream_task_ids': [], 'downstream_task_ids': []}, 'is_paused_upon_creation': False, '_dag_id': 'simple_dag', 'doc_md': '### DAG Tutorial Documentation', 'fileloc': None, '_processor_dags_folder': f'{repo_root}/tests/dags', 'tasks': [{'task_id': 'bash_task', 'owner': 'airflow', 'retries': 1, 'retry_delay': 300.0, 'max_retry_delay': 600.0, 'sla': 100.0, 'downstream_task_ids': [], '_is_empty': False, 'ui_color': '#f0ede4', 'ui_fgcolor': '#000', 'template_ext': ['.sh', '.bash'], 'template_fields': ['bash_command', 'env'], 'template_fields_renderers': {'bash_command': 'bash', 'env': 'json'}, 'bash_command': 'echo {{ task.task_id }}', '_task_type': 'BashOperator', '_task_module': 'airflow.operators.bash', 'pool': 'default_pool', 'is_setup': False, 'is_teardown': False, 'on_failure_fail_dagrun': False, 'executor_config': {'__type': 'dict', '__var': {'pod_override': {'__type': 'k8s.V1Pod', '__var': PodGenerator.serialize_pod(executor_config_pod)}}}, 'doc_md': '### Task Tutorial Documentation', '_log_config_logger_name': 'airflow.task.operators'}, {'task_id': 'custom_task', 'retries': 1, 'retry_delay': 300.0, 'max_retry_delay': 600.0, 'sla': 100.0, 'downstream_task_ids': [], '_is_empty': False, '_operator_extra_links': [{'tests.test_utils.mock_operators.CustomOpLink': {}}], 'ui_color': '#fff', 'ui_fgcolor': '#000', 'template_ext': [], 'template_fields': ['bash_command'], 'template_fields_renderers': {}, '_task_type': 'CustomOperator', '_operator_name': '@custom', '_task_module': 'tests.test_utils.mock_operators', 'pool': 'default_pool', 'is_setup': False, 'is_teardown': False, 'on_failure_fail_dagrun': False, '_log_config_logger_name': 'airflow.task.operators'}], 'schedule_interval': {'__type': 'timedelta', '__var': 86400.0}, 'dataset_triggers': [], 'timezone': 'UTC', '_access_control': {'__type': 'dict', '__var': {'test_role': {'__type': 'set', '__var': [permissions.ACTION_CAN_READ, permissions.ACTION_CAN_EDIT]}}}, 'edge_info': {}, 'dag_dependencies': [], 'params': {}}}
ROOT_FOLDER = os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir))
CUSTOM_TIMETABLE_SERIALIZED = {'__type': 'tests.test_utils.timetables.CustomSerializationTimetable', '__var': {'value': 'foo'}}

def make_example_dags(module_path):
    if False:
        print('Hello World!')
    'Loads DAGs from a module for test.'
    dagbag = DagBag(module_path)
    return dagbag.dags

def make_simple_dag():
    if False:
        return 10
    'Make very simple DAG to verify serialization result.'
    with DAG(dag_id='simple_dag', default_args={'retries': 1, 'retry_delay': timedelta(minutes=5), 'max_retry_delay': timedelta(minutes=10), 'depends_on_past': False, 'sla': timedelta(seconds=100)}, start_date=datetime(2019, 8, 1), is_paused_upon_creation=False, access_control={'test_role': {permissions.ACTION_CAN_READ, permissions.ACTION_CAN_EDIT}}, doc_md='### DAG Tutorial Documentation') as dag:
        CustomOperator(task_id='custom_task')
        BashOperator(task_id='bash_task', bash_command='echo {{ task.task_id }}', owner='airflow', executor_config={'pod_override': executor_config_pod}, doc_md='### Task Tutorial Documentation')
        return {'simple_dag': dag}

def make_user_defined_macro_filter_dag():
    if False:
        for i in range(10):
            print('nop')
    'Make DAGs with user defined macros and filters using locally defined methods.\n\n    For Webserver, we do not include ``user_defined_macros`` & ``user_defined_filters``.\n\n    The examples here test:\n        (1) functions can be successfully displayed on UI;\n        (2) templates with function macros have been rendered before serialization.\n    '

    def compute_next_execution_date(dag, execution_date):
        if False:
            i = 10
            return i + 15
        return dag.following_schedule(execution_date)
    default_args = {'start_date': datetime(2019, 7, 10)}
    dag = DAG('user_defined_macro_filter_dag', default_args=default_args, user_defined_macros={'next_execution_date': compute_next_execution_date}, user_defined_filters={'hello': lambda name: f'Hello {name}'}, catchup=False)
    BashOperator(task_id='echo', bash_command='echo "{{ next_execution_date(dag, execution_date) }}"', dag=dag)
    return {dag.dag_id: dag}

def collect_dags(dag_folder=None):
    if False:
        for i in range(10):
            print('nop')
    'Collects DAGs to test.'
    dags = {}
    dags.update(make_simple_dag())
    dags.update(make_user_defined_macro_filter_dag())
    if dag_folder:
        if isinstance(dag_folder, (list, tuple)):
            patterns = dag_folder
        else:
            patterns = [dag_folder]
    else:
        patterns = ['airflow/example_dags', 'airflow/providers/*/example_dags', 'airflow/providers/*/*/example_dags', 'tests/system/providers/*/', 'tests/system/providers/*/*/']
    for pattern in patterns:
        for directory in glob(f'{ROOT_FOLDER}/{pattern}'):
            dags.update(make_example_dags(directory))
    dags = {dag_id: dag for (dag_id, dag) in dags.items() if not dag.is_subdag}
    return dags

def get_timetable_based_simple_dag(timetable):
    if False:
        while True:
            i = 10
    'Create a simple_dag variant that uses timetable instead of schedule_interval.'
    dag = collect_dags(['airflow/example_dags'])['simple_dag']
    dag.timetable = timetable
    dag.schedule_interval = timetable.summary
    return dag

def serialize_subprocess(queue, dag_folder):
    if False:
        print('Hello World!')
    'Validate pickle in a subprocess.'
    dags = collect_dags(dag_folder)
    for dag in dags.values():
        queue.put(SerializedDAG.to_json(dag))
    queue.put(None)

@pytest.fixture()
def timetable_plugin(monkeypatch):
    if False:
        for i in range(10):
            print('nop')
    'Patch plugins manager to always and only return our custom timetable.'
    from airflow import plugins_manager
    monkeypatch.setattr(plugins_manager, 'initialize_timetables_plugins', lambda : None)
    monkeypatch.setattr(plugins_manager, 'timetable_classes', {'tests.test_utils.timetables.CustomSerializationTimetable': CustomSerializationTimetable})

class TestStringifiedDAGs:
    """Unit tests for stringified DAGs."""

    def setup_method(self):
        if False:
            i = 10
            return i + 15
        self.backup_base_hook_get_connection = BaseHook.get_connection
        BaseHook.get_connection = mock.Mock(return_value=Connection(extra='{"project_id": "mock", "location": "mock", "instance": "mock", "database_type": "postgres", "use_proxy": "False", "use_ssl": "False"}'))
        self.maxDiff = None

    def teardown_method(self):
        if False:
            return 10
        BaseHook.get_connection = self.backup_base_hook_get_connection

    @pytest.mark.db_test
    def test_serialization(self):
        if False:
            return 10
        'Serialization and deserialization should work for every DAG and Operator.'
        dags = collect_dags()
        serialized_dags = {}
        for v in dags.values():
            dag = SerializedDAG.to_dict(v)
            SerializedDAG.validate_schema(dag)
            serialized_dags[v.dag_id] = dag
        (actual, expected) = self.prepare_ser_dags_for_comparison(actual=serialized_dags['simple_dag'], expected=serialized_simple_dag_ground_truth)
        assert actual == expected

    @pytest.mark.parametrize('timetable, serialized_timetable', [(cron_timetable('0 0 * * *'), {'__type': 'airflow.timetables.interval.CronDataIntervalTimetable', '__var': {'expression': '0 0 * * *', 'timezone': 'UTC'}}), (CustomSerializationTimetable('foo'), CUSTOM_TIMETABLE_SERIALIZED)])
    @pytest.mark.usefixtures('timetable_plugin')
    def test_dag_serialization_to_timetable(self, timetable, serialized_timetable):
        if False:
            for i in range(10):
                print('nop')
        'Verify a timetable-backed schedule_interval is excluded in serialization.'
        dag = get_timetable_based_simple_dag(timetable)
        serialized_dag = SerializedDAG.to_dict(dag)
        SerializedDAG.validate_schema(serialized_dag)
        expected = copy.deepcopy(serialized_simple_dag_ground_truth)
        del expected['dag']['schedule_interval']
        expected['dag']['timetable'] = serialized_timetable
        (actual, expected) = self.prepare_ser_dags_for_comparison(actual=serialized_dag, expected=expected)
        for task in actual['dag']['tasks']:
            for (k, v) in task.items():
                print(task['task_id'], k, v)
        assert actual == expected

    def test_dag_serialization_preserves_empty_access_roles(self):
        if False:
            print('Hello World!')
        'Verify that an explicitly empty access_control dict is preserved.'
        dag = collect_dags(['airflow/example_dags'])['simple_dag']
        dag.access_control = {}
        serialized_dag = SerializedDAG.to_dict(dag)
        SerializedDAG.validate_schema(serialized_dag)
        assert serialized_dag['dag']['_access_control'] == {'__type': 'dict', '__var': {}}

    def test_dag_serialization_unregistered_custom_timetable(self):
        if False:
            print('Hello World!')
        'Verify serialization fails without timetable registration.'
        dag = get_timetable_based_simple_dag(CustomSerializationTimetable('bar'))
        with pytest.raises(SerializationError) as ctx:
            SerializedDAG.to_dict(dag)
        message = "Failed to serialize DAG 'simple_dag': Timetable class 'tests.test_utils.timetables.CustomSerializationTimetable' is not registered or you have a top level database access that disrupted the session. Please check the airflow best practices documentation."
        assert str(ctx.value) == message

    def prepare_ser_dags_for_comparison(self, actual, expected):
        if False:
            for i in range(10):
                print('nop')
        'Verify serialized DAGs match the ground truth.'
        assert actual['dag']['fileloc'].split('/')[-1] == 'test_dag_serialization.py'
        actual['dag']['fileloc'] = None

        def sorted_serialized_dag(dag_dict: dict):
            if False:
                while True:
                    i = 10
            '\n            Sorts the "tasks" list and "access_control" permissions in the\n            serialised dag python dictionary. This is needed as the order of\n            items should not matter but assertEqual would fail if the order of\n            items changes in the dag dictionary\n            '
            dag_dict['dag']['tasks'] = sorted(dag_dict['dag']['tasks'], key=sorted)
            dag_dict['dag']['_access_control']['__var']['test_role']['__var'] = sorted(dag_dict['dag']['_access_control']['__var']['test_role']['__var'])
            return dag_dict
        actual = json.loads(json.dumps(sorted_serialized_dag(actual)))
        expected = json.loads(json.dumps(sorted_serialized_dag(expected)))
        return (actual, expected)

    def test_deserialization_across_process(self):
        if False:
            print('Hello World!')
        'A serialized DAG can be deserialized in another process.'
        queue = multiprocessing.Queue()
        proc = multiprocessing.Process(target=serialize_subprocess, args=(queue, 'airflow/example_dags'))
        proc.daemon = True
        proc.start()
        stringified_dags = {}
        while True:
            v = queue.get()
            if v is None:
                break
            dag = SerializedDAG.from_json(v)
            assert isinstance(dag, DAG)
            stringified_dags[dag.dag_id] = dag
        dags = collect_dags('airflow/example_dags')
        assert set(stringified_dags.keys()) == set(dags.keys())
        for dag_id in stringified_dags:
            self.validate_deserialized_dag(stringified_dags[dag_id], dags[dag_id])

    def test_roundtrip_provider_example_dags(self):
        if False:
            i = 10
            return i + 15
        dags = collect_dags(['airflow/providers/*/example_dags', 'airflow/providers/*/*/example_dags'])
        for dag in dags.values():
            serialized_dag = SerializedDAG.from_json(SerializedDAG.to_json(dag))
            self.validate_deserialized_dag(serialized_dag, dag)

    @pytest.mark.parametrize('timetable', [cron_timetable('0 0 * * *'), CustomSerializationTimetable('foo')])
    @pytest.mark.usefixtures('timetable_plugin')
    def test_dag_roundtrip_from_timetable(self, timetable):
        if False:
            while True:
                i = 10
        'Verify a timetable-backed serialization can be deserialized.'
        dag = get_timetable_based_simple_dag(timetable)
        roundtripped = SerializedDAG.from_json(SerializedDAG.to_json(dag))
        self.validate_deserialized_dag(roundtripped, dag)

    def validate_deserialized_dag(self, serialized_dag, dag):
        if False:
            print('Hello World!')
        '\n        Verify that all example DAGs work with DAG Serialization by\n        checking fields between Serialized Dags & non-Serialized Dags\n        '
        exclusion_list = {'timetable', 'timezone', 'default_args', '_task_group', 'params', '_processor_dags_folder'}
        fields_to_check = dag.get_serialized_fields() - exclusion_list
        for field in fields_to_check:
            assert getattr(serialized_dag, field) == getattr(dag, field), f'{dag.dag_id}.{field} does not match'
        assert dag._processor_dags_folder is None
        assert serialized_dag._processor_dags_folder == str(repo_root / 'tests/dags')
        if dag.default_args:
            for (k, v) in dag.default_args.items():
                if callable(v):
                    assert k in serialized_dag.default_args
                else:
                    assert v == serialized_dag.default_args[k], f'{dag.dag_id}.default_args[{k}] does not match'
        assert serialized_dag.timetable.summary == dag.timetable.summary
        assert serialized_dag.timetable.serialize() == dag.timetable.serialize()
        assert serialized_dag.timezone.name == dag.timezone.name
        for task_id in dag.task_ids:
            self.validate_deserialized_task(serialized_dag.get_task(task_id), dag.get_task(task_id))

    def validate_deserialized_task(self, serialized_task, task):
        if False:
            for i in range(10):
                print('nop')
        'Verify non-Airflow operators are casted to BaseOperator or MappedOperator.'
        assert not isinstance(task, SerializedBaseOperator)
        assert isinstance(task, (BaseOperator, MappedOperator))
        assert serialized_task.task_group
        if isinstance(task, BaseOperator):
            assert isinstance(serialized_task, SerializedBaseOperator)
            fields_to_check = task.get_serialized_fields() - {'_task_type', '_operator_name', 'subdag', '_log', 'template_ext', 'template_fields', 'on_failure_callback', 'on_success_callback', 'on_retry_callback', 'resources', 'on_failure_fail_dagrun'}
        else:
            assert isinstance(serialized_task, MappedOperator)
            fields_to_check = {f.name for f in attr.fields(MappedOperator)}
            fields_to_check -= {'dag', 'task_group', 'operator_extra_links', 'template_ext', 'template_fields', 'operator_class', 'partial_kwargs'}
        assert serialized_task.task_type == task.task_type
        assert set(serialized_task.template_ext) == set(task.template_ext)
        assert set(serialized_task.template_fields) == set(task.template_fields)
        assert serialized_task.upstream_task_ids == task.upstream_task_ids
        assert serialized_task.downstream_task_ids == task.downstream_task_ids
        for field in fields_to_check:
            assert getattr(serialized_task, field) == getattr(task, field), f'{task.dag.dag_id}.{task.task_id}.{field} does not match'
        if serialized_task.resources is None:
            assert task.resources is None or task.resources == []
        else:
            assert serialized_task.resources == task.resources
        if isinstance(task.params, ParamsDict) and isinstance(serialized_task.params, ParamsDict):
            assert serialized_task.params.dump() == task.params.dump()
        if isinstance(task, MappedOperator):
            serialized_task.operator_class['_task_type'] == type(task).__name__
            if isinstance(serialized_task.operator_class, DecoratedOperator):
                serialized_task.operator_class['_operator_name'] == task._operator_name
            default_partial_kwargs = BaseOperator.partial(task_id='_')._expand(EXPAND_INPUT_EMPTY, strict=False).partial_kwargs
            serialized_partial_kwargs = {**default_partial_kwargs, **serialized_task.partial_kwargs}
            original_partial_kwargs = {**default_partial_kwargs, **task.partial_kwargs}
            assert serialized_partial_kwargs == original_partial_kwargs
        if task.task_type == 'SubDagOperator':
            assert serialized_task.subdag is not None
            assert isinstance(serialized_task.subdag, DAG)
        else:
            assert serialized_task.subdag is None

    @pytest.mark.parametrize('dag_start_date, task_start_date, expected_task_start_date', [(datetime(2019, 8, 1, tzinfo=timezone.utc), None, datetime(2019, 8, 1, tzinfo=timezone.utc)), (datetime(2019, 8, 1, tzinfo=timezone.utc), datetime(2019, 8, 2, tzinfo=timezone.utc), datetime(2019, 8, 2, tzinfo=timezone.utc)), (datetime(2019, 8, 1, tzinfo=timezone.utc), datetime(2019, 7, 30, tzinfo=timezone.utc), datetime(2019, 8, 1, tzinfo=timezone.utc)), (pendulum.datetime(2019, 8, 1, tz='UTC'), None, pendulum.datetime(2019, 8, 1, tz='UTC'))])
    def test_deserialization_start_date(self, dag_start_date, task_start_date, expected_task_start_date):
        if False:
            i = 10
            return i + 15
        dag = DAG(dag_id='simple_dag', start_date=dag_start_date)
        BaseOperator(task_id='simple_task', dag=dag, start_date=task_start_date)
        serialized_dag = SerializedDAG.to_dict(dag)
        if not task_start_date or dag_start_date >= task_start_date:
            assert 'start_date' not in serialized_dag['dag']['tasks'][0]
        else:
            assert 'start_date' in serialized_dag['dag']['tasks'][0]
        dag = SerializedDAG.from_dict(serialized_dag)
        simple_task = dag.task_dict['simple_task']
        assert simple_task.start_date == expected_task_start_date

    def test_deserialization_with_dag_context(self):
        if False:
            return 10
        with DAG(dag_id='simple_dag', start_date=datetime(2019, 8, 1, tzinfo=timezone.utc)) as dag:
            BaseOperator(task_id='simple_task')
            SerializedDAG.to_dict(dag)

    @pytest.mark.parametrize('dag_end_date, task_end_date, expected_task_end_date', [(datetime(2019, 8, 1, tzinfo=timezone.utc), None, datetime(2019, 8, 1, tzinfo=timezone.utc)), (datetime(2019, 8, 1, tzinfo=timezone.utc), datetime(2019, 8, 2, tzinfo=timezone.utc), datetime(2019, 8, 1, tzinfo=timezone.utc)), (datetime(2019, 8, 1, tzinfo=timezone.utc), datetime(2019, 7, 30, tzinfo=timezone.utc), datetime(2019, 7, 30, tzinfo=timezone.utc))])
    def test_deserialization_end_date(self, dag_end_date, task_end_date, expected_task_end_date):
        if False:
            while True:
                i = 10
        dag = DAG(dag_id='simple_dag', start_date=datetime(2019, 8, 1), end_date=dag_end_date)
        BaseOperator(task_id='simple_task', dag=dag, end_date=task_end_date)
        serialized_dag = SerializedDAG.to_dict(dag)
        if not task_end_date or dag_end_date <= task_end_date:
            assert 'end_date' not in serialized_dag['dag']['tasks'][0]
        else:
            assert 'end_date' in serialized_dag['dag']['tasks'][0]
        dag = SerializedDAG.from_dict(serialized_dag)
        simple_task = dag.task_dict['simple_task']
        assert simple_task.end_date == expected_task_end_date

    @pytest.mark.parametrize('serialized_timetable, expected_timetable', [({'__type': 'airflow.timetables.simple.NullTimetable', '__var': {}}, NullTimetable()), ({'__type': 'airflow.timetables.interval.CronDataIntervalTimetable', '__var': {'expression': '@weekly', 'timezone': 'UTC'}}, cron_timetable('0 0 * * 0')), ({'__type': 'airflow.timetables.simple.OnceTimetable', '__var': {}}, OnceTimetable()), ({'__type': 'airflow.timetables.interval.DeltaDataIntervalTimetable', '__var': {'delta': 86400.0}}, delta_timetable(timedelta(days=1))), (CUSTOM_TIMETABLE_SERIALIZED, CustomSerializationTimetable('foo'))])
    @pytest.mark.usefixtures('timetable_plugin')
    def test_deserialization_timetable(self, serialized_timetable, expected_timetable):
        if False:
            return 10
        serialized = {'__version': 1, 'dag': {'default_args': {'__type': 'dict', '__var': {}}, '_dag_id': 'simple_dag', 'fileloc': __file__, 'tasks': [], 'timezone': 'UTC', 'timetable': serialized_timetable}}
        SerializedDAG.validate_schema(serialized)
        dag = SerializedDAG.from_dict(serialized)
        assert dag.timetable == expected_timetable

    def test_deserialization_timetable_unregistered(self):
        if False:
            while True:
                i = 10
        serialized = {'__version': 1, 'dag': {'default_args': {'__type': 'dict', '__var': {}}, '_dag_id': 'simple_dag', 'fileloc': __file__, 'tasks': [], 'timezone': 'UTC', 'timetable': CUSTOM_TIMETABLE_SERIALIZED}}
        SerializedDAG.validate_schema(serialized)
        with pytest.raises(ValueError) as ctx:
            SerializedDAG.from_dict(serialized)
        message = "Timetable class 'tests.test_utils.timetables.CustomSerializationTimetable' is not registered or you have a top level database access that disrupted the session. Please check the airflow best practices documentation."
        assert str(ctx.value) == message

    @pytest.mark.parametrize('serialized_schedule_interval, expected_timetable', [(None, NullTimetable()), ('@weekly', cron_timetable('0 0 * * 0')), ('@once', OnceTimetable()), ({'__type': 'timedelta', '__var': 86400.0}, delta_timetable(timedelta(days=1)))])
    def test_deserialization_schedule_interval(self, serialized_schedule_interval, expected_timetable):
        if False:
            i = 10
            return i + 15
        'Test DAGs serialized before 2.2 can be correctly deserialized.'
        serialized = {'__version': 1, 'dag': {'default_args': {'__type': 'dict', '__var': {}}, '_dag_id': 'simple_dag', 'fileloc': __file__, 'tasks': [], 'timezone': 'UTC', 'schedule_interval': serialized_schedule_interval}}
        SerializedDAG.validate_schema(serialized)
        dag = SerializedDAG.from_dict(serialized)
        assert dag.timetable == expected_timetable

    @pytest.mark.parametrize('val, expected', [(relativedelta(days=-1), {'__type': 'relativedelta', '__var': {'days': -1}}), (relativedelta(month=1, days=-1), {'__type': 'relativedelta', '__var': {'month': 1, 'days': -1}}), (relativedelta(weekday=FR), {'__type': 'relativedelta', '__var': {'weekday': [4]}}), (relativedelta(weekday=FR(2)), {'__type': 'relativedelta', '__var': {'weekday': [4, 2]}})])
    def test_roundtrip_relativedelta(self, val, expected):
        if False:
            print('Hello World!')
        serialized = SerializedDAG.serialize(val)
        assert serialized == expected
        round_tripped = SerializedDAG.deserialize(serialized)
        assert val == round_tripped

    @pytest.mark.parametrize('val, expected_val', [(None, {}), ({'param_1': 'value_1'}, {'param_1': 'value_1'}), ({'param_1': {1, 2, 3}}, {'param_1': {1, 2, 3}})])
    def test_dag_params_roundtrip(self, val, expected_val):
        if False:
            while True:
                i = 10
        '\n        Test that params work both on Serialized DAGs & Tasks\n        '
        dag = DAG(dag_id='simple_dag', params=val)
        BaseOperator(task_id='simple_task', dag=dag, start_date=datetime(2019, 8, 1))
        serialized_dag_json = SerializedDAG.to_json(dag)
        serialized_dag = json.loads(serialized_dag_json)
        assert 'params' in serialized_dag['dag']
        deserialized_dag = SerializedDAG.from_dict(serialized_dag)
        deserialized_simple_task = deserialized_dag.task_dict['simple_task']
        assert expected_val == deserialized_dag.params.dump()
        assert expected_val == deserialized_simple_task.params.dump()

    def test_invalid_params(self):
        if False:
            return 10
        '\n        Test to make sure that only native Param objects are being passed as dag or task params\n        '

        class S3Param(Param):

            def __init__(self, path: str):
                if False:
                    return 10
                schema = {'type': 'string', 'pattern': 's3:\\/\\/(.+?)\\/(.+)'}
                super().__init__(default=path, schema=schema)
        dag = DAG(dag_id='simple_dag', params={'path': S3Param('s3://my_bucket/my_path')})
        with pytest.raises(SerializationError):
            SerializedDAG.to_dict(dag)
        dag = DAG(dag_id='simple_dag')
        BaseOperator(task_id='simple_task', dag=dag, start_date=datetime(2019, 8, 1), params={'path': S3Param('s3://my_bucket/my_path')})

    @pytest.mark.parametrize('param', [Param('my value', description='hello', schema={'type': 'string'}), Param('my value', description='hello'), Param(None, description=None), Param([True], type='array', items={'type': 'boolean'}), Param()])
    def test_full_param_roundtrip(self, param: Param):
        if False:
            i = 10
            return i + 15
        '\n        Test to make sure that only native Param objects are being passed as dag or task params\n        '
        dag = DAG(dag_id='simple_dag', schedule=None, params={'my_param': param})
        serialized_json = SerializedDAG.to_json(dag)
        serialized = json.loads(serialized_json)
        SerializedDAG.validate_schema(serialized)
        dag = SerializedDAG.from_dict(serialized)
        assert dag.params.get_param('my_param').value == param.value
        observed_param = dag.params.get_param('my_param')
        assert isinstance(observed_param, Param)
        assert observed_param.description == param.description
        assert observed_param.schema == param.schema

    @pytest.mark.parametrize('val, expected_val', [(None, {}), ({'param_1': 'value_1'}, {'param_1': 'value_1'}), ({'param_1': {1, 2, 3}}, {'param_1': {1, 2, 3}})])
    def test_task_params_roundtrip(self, val, expected_val):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test that params work both on Serialized DAGs & Tasks\n        '
        dag = DAG(dag_id='simple_dag')
        BaseOperator(task_id='simple_task', dag=dag, params=val, start_date=datetime(2019, 8, 1))
        serialized_dag = SerializedDAG.to_dict(dag)
        if val:
            assert 'params' in serialized_dag['dag']['tasks'][0]
        else:
            assert 'params' not in serialized_dag['dag']['tasks'][0]
        deserialized_dag = SerializedDAG.from_dict(serialized_dag)
        deserialized_simple_task = deserialized_dag.task_dict['simple_task']
        assert expected_val == deserialized_simple_task.params.dump()

    @pytest.mark.db_test
    @pytest.mark.parametrize(('bash_command', 'serialized_links', 'links'), [pytest.param('true', [{'tests.test_utils.mock_operators.CustomOpLink': {}}], {'Google Custom': 'http://google.com/custom_base_link?search=true'}, id='non-indexed-link'), pytest.param(['echo', 'true'], [{'tests.test_utils.mock_operators.CustomBaseIndexOpLink': {'index': 0}}, {'tests.test_utils.mock_operators.CustomBaseIndexOpLink': {'index': 1}}], {'BigQuery Console #1': 'https://console.cloud.google.com/bigquery?j=echo', 'BigQuery Console #2': 'https://console.cloud.google.com/bigquery?j=true'}, id='multiple-indexed-links')])
    def test_extra_serialized_field_and_operator_links(self, bash_command, serialized_links, links, dag_maker):
        if False:
            print('Hello World!')
        '\n        Assert extra field exists & OperatorLinks defined in Plugins and inbuilt Operator Links.\n\n        This tests also depends on GoogleLink() registered as a plugin\n        in tests/plugins/test_plugin.py\n\n        The function tests that if extra operator links are registered in plugin\n        in ``operator_extra_links`` and the same is also defined in\n        the Operator in ``BaseOperator.operator_extra_links``, it has the correct\n        extra link.\n\n        If CustomOperator is called with a string argument for bash_command it\n        has a single link, if called with an array it has one link per element.\n        We use this to test the serialization of link data.\n        '
        test_date = timezone.DateTime(2019, 8, 1, tzinfo=timezone.utc)
        with dag_maker(dag_id='simple_dag', start_date=test_date) as dag:
            CustomOperator(task_id='simple_task', bash_command=bash_command)
        serialized_dag = SerializedDAG.to_dict(dag)
        assert 'bash_command' in serialized_dag['dag']['tasks'][0]
        dag = SerializedDAG.from_dict(serialized_dag)
        simple_task = dag.task_dict['simple_task']
        assert getattr(simple_task, 'bash_command') == bash_command
        assert serialized_dag['dag']['tasks'][0]['_operator_extra_links'] == serialized_links
        assert simple_task.extra_links == sorted({*links, 'airflow', 'github', 'google'})
        dr = dag_maker.create_dagrun(execution_date=test_date)
        (ti,) = dr.task_instances
        XCom.set(key='search_query', value=bash_command, task_id=simple_task.task_id, dag_id=simple_task.dag_id, run_id=dr.run_id)
        for (name, expected) in links.items():
            link = simple_task.get_extra_links(ti, name)
            assert link == expected
        link = simple_task.get_extra_links(ti, GoogleLink.name)
        assert 'https://www.google.com' == link

    @pytest.mark.db_test
    def test_extra_operator_links_logs_error_for_non_registered_extra_links(self, caplog):
        if False:
            while True:
                i = 10
        '\n        Assert OperatorLinks not registered via Plugins and if it is not an inbuilt Operator Link,\n        it can still deserialize the DAG (does not error) but just logs an error\n        '

        class TaskStateLink(BaseOperatorLink):
            """OperatorLink not registered via Plugins nor a built-in OperatorLink"""
            name = 'My Link'

            def get_link(self, operator, *, ti_key):
                if False:
                    while True:
                        i = 10
                return 'https://www.google.com'

        class MyOperator(BaseOperator):
            """Just a EmptyOperator using above defined Extra Operator Link"""
            operator_extra_links = [TaskStateLink()]

            def execute(self, context: Context):
                if False:
                    return 10
                pass
        with DAG(dag_id='simple_dag', start_date=datetime(2019, 8, 1)) as dag:
            MyOperator(task_id='blah')
        serialized_dag = SerializedDAG.to_dict(dag)
        with caplog.at_level('ERROR', logger='airflow.serialization.serialized_objects'):
            SerializedDAG.from_dict(serialized_dag)
        expected_err_msg = "Operator Link class 'tests.serialization.test_dag_serialization.TaskStateLink' not registered"
        assert expected_err_msg in caplog.text

    class ClassWithCustomAttributes:
        """
        Class for testing purpose: allows to create objects with custom attributes in one single statement.
        """

        def __init__(self, **kwargs):
            if False:
                print('Hello World!')
            for (key, value) in kwargs.items():
                setattr(self, key, value)

        def __str__(self):
            if False:
                return 10
            return f'{self.__class__.__name__}({str(self.__dict__)})'

        def __repr__(self):
            if False:
                return 10
            return self.__str__()

        def __eq__(self, other):
            if False:
                print('Hello World!')
            return self.__dict__ == other.__dict__

        def __ne__(self, other):
            if False:
                i = 10
                return i + 15
            return not self.__eq__(other)

    @pytest.mark.parametrize('templated_field, expected_field', [(None, None), ([], []), ({}, {}), ('{{ task.task_id }}', '{{ task.task_id }}'), ['{{ task.task_id }}', '{{ task.task_id }}'], ({'foo': '{{ task.task_id }}'}, {'foo': '{{ task.task_id }}'}), ({'foo': {'bar': '{{ task.task_id }}'}}, {'foo': {'bar': '{{ task.task_id }}'}}), ([{'foo1': {'bar': '{{ task.task_id }}'}}, {'foo2': {'bar': '{{ task.task_id }}'}}], [{'foo1': {'bar': '{{ task.task_id }}'}}, {'foo2': {'bar': '{{ task.task_id }}'}}]), ({'foo': {'bar': {'{{ task.task_id }}': ['sar']}}}, {'foo': {'bar': {'{{ task.task_id }}': ['sar']}}}), (ClassWithCustomAttributes(att1='{{ task.task_id }}', att2='{{ task.task_id }}', template_fields=['att1']), "ClassWithCustomAttributes({'att1': '{{ task.task_id }}', 'att2': '{{ task.task_id }}', 'template_fields': ['att1']})"), (ClassWithCustomAttributes(nested1=ClassWithCustomAttributes(att1='{{ task.task_id }}', att2='{{ task.task_id }}', template_fields=['att1']), nested2=ClassWithCustomAttributes(att3='{{ task.task_id }}', att4='{{ task.task_id }}', template_fields=['att3']), template_fields=['nested1']), "ClassWithCustomAttributes({'nested1': ClassWithCustomAttributes({'att1': '{{ task.task_id }}', 'att2': '{{ task.task_id }}', 'template_fields': ['att1']}), 'nested2': ClassWithCustomAttributes({'att3': '{{ task.task_id }}', 'att4': '{{ task.task_id }}', 'template_fields': ['att3']}), 'template_fields': ['nested1']})")])
    def test_templated_fields_exist_in_serialized_dag(self, templated_field, expected_field):
        if False:
            while True:
                i = 10
        '\n        Test that templated_fields exists for all Operators in Serialized DAG\n\n        Since we don\'t want to inflate arbitrary python objects (it poses a RCE/security risk etc.)\n        we want check that non-"basic" objects are turned in to strings after deserializing.\n        '
        dag = DAG('test_serialized_template_fields', start_date=datetime(2019, 8, 1))
        with dag:
            BashOperator(task_id='test', bash_command=templated_field)
        serialized_dag = SerializedDAG.to_dict(dag)
        deserialized_dag = SerializedDAG.from_dict(serialized_dag)
        deserialized_test_task = deserialized_dag.task_dict['test']
        assert expected_field == getattr(deserialized_test_task, 'bash_command')

    def test_dag_serialized_fields_with_schema(self):
        if False:
            print('Hello World!')
        '\n        Additional Properties are disabled on DAGs. This test verifies that all the\n        keys in DAG.get_serialized_fields are listed in Schema definition.\n        '
        dag_schema: dict = load_dag_schema_dict()['definitions']['dag']['properties']
        ignored_keys: set = {'is_subdag', 'tasks', 'has_on_success_callback', 'has_on_failure_callback', 'dag_dependencies', 'params'}
        keys_for_backwards_compat: set = {'_concurrency'}
        dag_params: set = set(dag_schema.keys()) - ignored_keys - keys_for_backwards_compat
        assert set(DAG.get_serialized_fields()) == dag_params

    def test_operator_subclass_changing_base_defaults(self):
        if False:
            i = 10
            return i + 15
        assert BaseOperator(task_id='dummy').do_xcom_push is True, "Precondition check! If this fails the test won't make sense"

        class MyOperator(BaseOperator):

            def __init__(self, do_xcom_push=False, **kwargs):
                if False:
                    print('Hello World!')
                super().__init__(**kwargs)
                self.do_xcom_push = do_xcom_push
        op = MyOperator(task_id='dummy')
        assert op.do_xcom_push is False
        blob = SerializedBaseOperator.serialize_operator(op)
        serialized_op = SerializedBaseOperator.deserialize_operator(blob)
        assert serialized_op.do_xcom_push is False

    def test_no_new_fields_added_to_base_operator(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        This test verifies that there are no new fields added to BaseOperator. And reminds that\n        tests should be added for it.\n        '
        base_operator = BaseOperator(task_id='10')
        fields = {k: v for (k, v) in vars(base_operator).items() if k in BaseOperator.get_serialized_fields()}
        assert fields == {'_logger_name': None, '_log_config_logger_name': 'airflow.task.operators', '_post_execute_hook': None, '_pre_execute_hook': None, 'depends_on_past': False, 'do_xcom_push': True, 'doc': None, 'doc_json': None, 'doc_md': None, 'doc_rst': None, 'doc_yaml': None, 'downstream_task_ids': set(), 'email': None, 'email_on_failure': True, 'email_on_retry': True, 'execution_timeout': None, 'executor_config': {}, 'ignore_first_depends_on_past': True, 'inlets': [], 'max_active_tis_per_dag': None, 'max_active_tis_per_dagrun': None, 'max_retry_delay': None, 'on_execute_callback': None, 'on_failure_callback': None, 'on_retry_callback': None, 'on_success_callback': None, 'outlets': [], 'owner': 'airflow', 'params': {}, 'pool': 'default_pool', 'pool_slots': 1, 'priority_weight': 1, 'queue': 'default', 'resources': None, 'retries': 0, 'retry_delay': timedelta(0, 300), 'retry_exponential_backoff': False, 'run_as_user': None, 'sla': None, 'task_id': '10', 'trigger_rule': 'all_success', 'wait_for_downstream': False, 'wait_for_past_depends_before_skipping': False, 'weight_rule': 'downstream'}, '\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n     ACTION NEEDED! PLEASE READ THIS CAREFULLY AND CORRECT TESTS CAREFULLY\n\n Some fields were added to the BaseOperator! Please add them to the list above and make sure that\n you add support for DAG serialization - you should add the field to\n `airflow/serialization/schema.json` - they should have correct type defined there.\n\n Note that we do not support versioning yet so you should only add optional fields to BaseOperator.\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n                         '

    def test_operator_deserialize_old_names(self):
        if False:
            print('Hello World!')
        blob = {'task_id': 'custom_task', '_downstream_task_ids': ['foo'], 'template_ext': [], 'template_fields': ['bash_command'], 'template_fields_renderers': {}, '_task_type': 'CustomOperator', '_task_module': 'tests.test_utils.mock_operators', 'pool': 'default_pool', 'ui_color': '#fff', 'ui_fgcolor': '#000'}
        SerializedDAG._json_schema.validate(blob, _schema=load_dag_schema_dict()['definitions']['operator'])
        serialized_op = SerializedBaseOperator.deserialize_operator(blob)
        assert serialized_op.downstream_task_ids == {'foo'}

    def test_task_resources(self):
        if False:
            return 10
        '\n        Test task resources serialization/deserialization.\n        '
        from airflow.operators.empty import EmptyOperator
        execution_date = datetime(2020, 1, 1)
        task_id = 'task1'
        with DAG('test_task_resources', start_date=execution_date) as dag:
            task = EmptyOperator(task_id=task_id, resources={'cpus': 0.1, 'ram': 2048})
        SerializedDAG.validate_schema(SerializedDAG.to_dict(dag))
        json_dag = SerializedDAG.from_json(SerializedDAG.to_json(dag))
        deserialized_task = json_dag.get_task(task_id)
        assert deserialized_task.resources == task.resources
        assert isinstance(deserialized_task.resources, Resources)

    def test_task_group_serialization(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test TaskGroup serialization/deserialization.\n        '
        execution_date = datetime(2020, 1, 1)
        with DAG('test_task_group_serialization', start_date=execution_date) as dag:
            task1 = EmptyOperator(task_id='task1')
            with TaskGroup('group234') as group234:
                _ = EmptyOperator(task_id='task2')
                with TaskGroup('group34') as group34:
                    _ = EmptyOperator(task_id='task3')
                    _ = EmptyOperator(task_id='task4')
            task5 = EmptyOperator(task_id='task5')
            task1 >> group234
            group34 >> task5
        dag_dict = SerializedDAG.to_dict(dag)
        SerializedDAG.validate_schema(dag_dict)
        json_dag = SerializedDAG.from_json(SerializedDAG.to_json(dag))
        self.validate_deserialized_dag(json_dag, dag)
        serialized_dag = SerializedDAG.deserialize_dag(SerializedDAG.serialize_dag(dag))
        assert serialized_dag.task_group.children
        assert serialized_dag.task_group.children.keys() == dag.task_group.children.keys()

        def check_task_group(node):
            if False:
                i = 10
                return i + 15
            assert node.dag is serialized_dag
            try:
                children = node.children.values()
            except AttributeError:
                expected_serialized = SerializedBaseOperator.serialize_operator(dag.get_task(node.task_id))
                expected_deserialized = SerializedBaseOperator.deserialize_operator(expected_serialized)
                expected_dict = SerializedBaseOperator.serialize_operator(expected_deserialized)
                assert node
                assert SerializedBaseOperator.serialize_operator(node) == expected_dict
                return
            for child in children:
                check_task_group(child)
        check_task_group(serialized_dag.task_group)

    @staticmethod
    def assert_taskgroup_children(se_task_group, dag_task_group, expected_children):
        if False:
            return 10
        assert se_task_group.children.keys() == dag_task_group.children.keys() == expected_children

    @staticmethod
    def assert_task_is_setup_teardown(task, is_setup: bool=False, is_teardown: bool=False):
        if False:
            for i in range(10):
                print('nop')
        assert task.is_setup == is_setup
        assert task.is_teardown == is_teardown

    def test_setup_teardown_tasks(self):
        if False:
            while True:
                i = 10
        '\n        Test setup and teardown task serialization/deserialization.\n        '
        execution_date = datetime(2020, 1, 1)
        with DAG('test_task_group_setup_teardown_tasks', start_date=execution_date) as dag:
            EmptyOperator(task_id='setup').as_setup()
            EmptyOperator(task_id='teardown').as_teardown()
            with TaskGroup('group1'):
                EmptyOperator(task_id='setup1').as_setup()
                EmptyOperator(task_id='task1')
                EmptyOperator(task_id='teardown1').as_teardown()
                with TaskGroup('group2'):
                    EmptyOperator(task_id='setup2').as_setup()
                    EmptyOperator(task_id='task2')
                    EmptyOperator(task_id='teardown2').as_teardown()
        dag_dict = SerializedDAG.to_dict(dag)
        SerializedDAG.validate_schema(dag_dict)
        json_dag = SerializedDAG.from_json(SerializedDAG.to_json(dag))
        self.validate_deserialized_dag(json_dag, dag)
        serialized_dag = SerializedDAG.deserialize_dag(SerializedDAG.serialize_dag(dag))
        self.assert_taskgroup_children(serialized_dag.task_group, dag.task_group, {'setup', 'teardown', 'group1'})
        self.assert_task_is_setup_teardown(serialized_dag.task_group.children['setup'], is_setup=True)
        self.assert_task_is_setup_teardown(serialized_dag.task_group.children['teardown'], is_teardown=True)
        se_first_group = serialized_dag.task_group.children['group1']
        dag_first_group = dag.task_group.children['group1']
        self.assert_taskgroup_children(se_first_group, dag_first_group, {'group1.setup1', 'group1.task1', 'group1.group2', 'group1.teardown1'})
        self.assert_task_is_setup_teardown(se_first_group.children['group1.setup1'], is_setup=True)
        self.assert_task_is_setup_teardown(se_first_group.children['group1.task1'])
        self.assert_task_is_setup_teardown(se_first_group.children['group1.teardown1'], is_teardown=True)
        se_second_group = se_first_group.children['group1.group2']
        dag_second_group = dag_first_group.children['group1.group2']
        self.assert_taskgroup_children(se_second_group, dag_second_group, {'group1.group2.setup2', 'group1.group2.task2', 'group1.group2.teardown2'})
        self.assert_task_is_setup_teardown(se_second_group.children['group1.group2.setup2'], is_setup=True)
        self.assert_task_is_setup_teardown(se_second_group.children['group1.group2.task2'])
        self.assert_task_is_setup_teardown(se_second_group.children['group1.group2.teardown2'], is_teardown=True)

    @pytest.mark.db_test
    def test_teardown_task_on_failure_fail_dagrun_serialization(self, dag_maker):
        if False:
            for i in range(10):
                print('nop')
        with dag_maker() as dag:

            @teardown(on_failure_fail_dagrun=True)
            def mytask():
                if False:
                    i = 10
                    return i + 15
                print(1)
            mytask()
        dag_dict = SerializedDAG.to_dict(dag)
        SerializedDAG.validate_schema(dag_dict)
        json_dag = SerializedDAG.from_json(SerializedDAG.to_json(dag))
        self.validate_deserialized_dag(json_dag, dag)
        serialized_dag = SerializedDAG.deserialize_dag(SerializedDAG.serialize_dag(dag))
        task = serialized_dag.task_group.children['mytask']
        assert task.is_teardown is True
        assert task.on_failure_fail_dagrun is True

    @pytest.mark.db_test
    def test_teardown_mapped_serialization(self, dag_maker):
        if False:
            return 10
        with dag_maker() as dag:

            @teardown(on_failure_fail_dagrun=True)
            def mytask(val=None):
                if False:
                    while True:
                        i = 10
                print(1)
            mytask.expand(val=[1, 2, 3])
        task = dag.task_group.children['mytask']
        assert task.partial_kwargs['is_teardown'] is True
        assert task.partial_kwargs['on_failure_fail_dagrun'] is True
        dag_dict = SerializedDAG.to_dict(dag)
        SerializedDAG.validate_schema(dag_dict)
        json_dag = SerializedDAG.from_json(SerializedDAG.to_json(dag))
        self.validate_deserialized_dag(json_dag, dag)
        serialized_dag = SerializedDAG.deserialize_dag(SerializedDAG.serialize_dag(dag))
        task = serialized_dag.task_group.children['mytask']
        assert task.partial_kwargs['is_teardown'] is True
        assert task.partial_kwargs['on_failure_fail_dagrun'] is True

    @pytest.mark.db_test
    def test_deps_sorted(self):
        if False:
            print('Hello World!')
        '\n        Tests serialize_operator, make sure the deps is in order\n        '
        from airflow.operators.empty import EmptyOperator
        from airflow.sensors.external_task import ExternalTaskSensor
        execution_date = datetime(2020, 1, 1)
        with DAG(dag_id='test_deps_sorted', start_date=execution_date) as dag:
            task1 = ExternalTaskSensor(task_id='task1', external_dag_id='external_dag_id', mode='reschedule')
            task2 = EmptyOperator(task_id='task2')
            task1 >> task2
        serialize_op = SerializedBaseOperator.serialize_operator(dag.task_dict['task1'])
        deps = serialize_op['deps']
        assert deps == ['airflow.ti_deps.deps.not_in_retry_period_dep.NotInRetryPeriodDep', 'airflow.ti_deps.deps.not_previously_skipped_dep.NotPreviouslySkippedDep', 'airflow.ti_deps.deps.prev_dagrun_dep.PrevDagrunDep', 'airflow.ti_deps.deps.ready_to_reschedule.ReadyToRescheduleDep', 'airflow.ti_deps.deps.trigger_rule_dep.TriggerRuleDep']

    def test_error_on_unregistered_ti_dep_serialization(self):
        if False:
            return 10

        class DummyTriggerRule(BaseTIDep):
            pass

        class DummyTask(BaseOperator):
            deps = frozenset([*BaseOperator.deps, DummyTriggerRule()])
        execution_date = datetime(2020, 1, 1)
        with DAG(dag_id='test_error_on_unregistered_ti_dep_serialization', start_date=execution_date) as dag:
            DummyTask(task_id='task1')
        with pytest.raises(SerializationError):
            SerializedBaseOperator.serialize_operator(dag.task_dict['task1'])

    def test_error_on_unregistered_ti_dep_deserialization(self):
        if False:
            for i in range(10):
                print('nop')
        from airflow.operators.empty import EmptyOperator
        with DAG('test_error_on_unregistered_ti_dep_deserialization', start_date=datetime(2019, 8, 1)) as dag:
            EmptyOperator(task_id='task1')
        serialize_op = SerializedBaseOperator.serialize_operator(dag.task_dict['task1'])
        serialize_op['deps'] = ['airflow.ti_deps.deps.not_in_retry_period_dep.NotInRetryPeriodDep', 'test_plugin.NotATriggerRule']
        with pytest.raises(SerializationError):
            SerializedBaseOperator.deserialize_operator(serialize_op)

    @pytest.mark.db_test
    def test_serialize_and_deserialize_custom_ti_deps(self):
        if False:
            for i in range(10):
                print('nop')
        from test_plugin import CustomTestTriggerRule

        class DummyTask(BaseOperator):
            deps = frozenset([*BaseOperator.deps, CustomTestTriggerRule()])
        execution_date = datetime(2020, 1, 1)
        with DAG(dag_id='test_serialize_custom_ti_deps', start_date=execution_date) as dag:
            DummyTask(task_id='task1')
        serialize_op = SerializedBaseOperator.serialize_operator(dag.task_dict['task1'])
        assert serialize_op['deps'] == ['airflow.ti_deps.deps.not_in_retry_period_dep.NotInRetryPeriodDep', 'airflow.ti_deps.deps.not_previously_skipped_dep.NotPreviouslySkippedDep', 'airflow.ti_deps.deps.prev_dagrun_dep.PrevDagrunDep', 'airflow.ti_deps.deps.trigger_rule_dep.TriggerRuleDep', 'test_plugin.CustomTestTriggerRule']
        op = SerializedBaseOperator.deserialize_operator(serialize_op)
        assert sorted((str(dep) for dep in op.deps)) == ['<TIDep(CustomTestTriggerRule)>', '<TIDep(Not In Retry Period)>', '<TIDep(Not Previously Skipped)>', '<TIDep(Previous Dagrun State)>', '<TIDep(Trigger Rule)>']

    def test_serialize_mapped_outlets(self):
        if False:
            for i in range(10):
                print('nop')
        with DAG(dag_id='d', start_date=datetime.now()):
            op = MockOperator.partial(task_id='x').expand(arg1=[1, 2])
        assert op.inlets == []
        assert op.outlets == []
        serialized = SerializedBaseOperator.serialize_mapped_operator(op)
        assert 'inlets' not in serialized
        assert 'outlets' not in serialized
        round_tripped = SerializedBaseOperator.deserialize_operator(serialized)
        assert isinstance(round_tripped, MappedOperator)
        assert round_tripped.inlets == []
        assert round_tripped.outlets == []

    @pytest.mark.db_test
    def test_derived_dag_deps_sensor(self):
        if False:
            print('Hello World!')
        '\n        Tests DAG dependency detection for sensors, including derived classes\n        '
        from airflow.operators.empty import EmptyOperator
        from airflow.sensors.external_task import ExternalTaskSensor

        class DerivedSensor(ExternalTaskSensor):
            pass
        execution_date = datetime(2020, 1, 1)
        for class_ in [ExternalTaskSensor, DerivedSensor]:
            with DAG(dag_id='test_derived_dag_deps_sensor', start_date=execution_date) as dag:
                task1 = class_(task_id='task1', external_dag_id='external_dag_id', mode='reschedule')
                task2 = EmptyOperator(task_id='task2')
                task1 >> task2
            dag = SerializedDAG.to_dict(dag)
            assert dag['dag']['dag_dependencies'] == [{'source': 'external_dag_id', 'target': 'test_derived_dag_deps_sensor', 'dependency_type': 'sensor', 'dependency_id': 'task1'}]

    @pytest.mark.db_test
    @conf_vars({('scheduler', 'dependency_detector'): 'tests.serialization.test_dag_serialization.CustomDependencyDetector'})
    def test_custom_dep_detector(self):
        if False:
            i = 10
            return i + 15
        '\n        Prior to deprecation of custom dependency detector, the return type was DagDependency | None.\n        This class verifies that custom dependency detector classes which assume that return type will still\n        work until support for them is removed in 3.0.\n\n        TODO: remove in Airflow 3.0\n        '
        from airflow.sensors.external_task import ExternalTaskSensor
        execution_date = datetime(2020, 1, 1)
        with DAG(dag_id='test', start_date=execution_date) as dag:
            ExternalTaskSensor(task_id='task1', external_dag_id='external_dag_id', mode='reschedule')
            CustomDepOperator(task_id='hello', bash_command='hi')
            dag = SerializedDAG.to_dict(dag)
            assert sorted(dag['dag']['dag_dependencies'], key=lambda x: tuple(x.values())) == sorted([{'source': 'external_dag_id', 'target': 'test', 'dependency_type': 'sensor', 'dependency_id': 'task1'}, {'source': 'test', 'target': 'nothing', 'dependency_type': 'abc', 'dependency_id': 'hello'}], key=lambda x: tuple(x.values()))

    @pytest.mark.db_test
    def test_dag_deps_datasets(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Check that dag_dependencies node is populated correctly for a DAG with datasets.\n        '
        from airflow.sensors.external_task import ExternalTaskSensor
        d1 = Dataset('d1')
        d2 = Dataset('d2')
        d3 = Dataset('d3')
        d4 = Dataset('d4')
        execution_date = datetime(2020, 1, 1)
        with DAG(dag_id='test', start_date=execution_date, schedule=[d1]) as dag:
            ExternalTaskSensor(task_id='task1', external_dag_id='external_dag_id', mode='reschedule')
            BashOperator(task_id='dataset_writer', bash_command='echo hello', outlets=[d2, d3])

            @dag.task(outlets=[d4])
            def other_dataset_writer(x):
                if False:
                    return 10
                pass
            other_dataset_writer.expand(x=[1, 2])
        dag = SerializedDAG.to_dict(dag)
        actual = sorted(dag['dag']['dag_dependencies'], key=lambda x: tuple(x.values()))
        expected = sorted([{'source': 'test', 'target': 'dataset', 'dependency_type': 'dataset', 'dependency_id': 'd4'}, {'source': 'external_dag_id', 'target': 'test', 'dependency_type': 'sensor', 'dependency_id': 'task1'}, {'source': 'test', 'target': 'dataset', 'dependency_type': 'dataset', 'dependency_id': 'd3'}, {'source': 'test', 'target': 'dataset', 'dependency_type': 'dataset', 'dependency_id': 'd2'}, {'source': 'dataset', 'target': 'test', 'dependency_type': 'dataset', 'dependency_id': 'd1'}], key=lambda x: tuple(x.values()))
        assert actual == expected

    def test_derived_dag_deps_operator(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Tests DAG dependency detection for operators, including derived classes\n        '
        from airflow.operators.empty import EmptyOperator
        from airflow.operators.trigger_dagrun import TriggerDagRunOperator

        class DerivedOperator(TriggerDagRunOperator):
            pass
        execution_date = datetime(2020, 1, 1)
        for class_ in [TriggerDagRunOperator, DerivedOperator]:
            with DAG(dag_id='test_derived_dag_deps_trigger', start_date=execution_date) as dag:
                task1 = EmptyOperator(task_id='task1')
                task2 = class_(task_id='task2', trigger_dag_id='trigger_dag_id')
                task1 >> task2
            dag = SerializedDAG.to_dict(dag)
            assert dag['dag']['dag_dependencies'] == [{'source': 'test_derived_dag_deps_trigger', 'target': 'trigger_dag_id', 'dependency_type': 'trigger', 'dependency_id': 'task2'}]

    def test_task_group_sorted(self):
        if False:
            print('Hello World!')
        '\n        Tests serialize_task_group, make sure the list is in order\n        '
        from airflow.operators.empty import EmptyOperator
        from airflow.serialization.serialized_objects import TaskGroupSerialization
        '\n                    start\n                    ╱  ╲\n                  ╱      ╲\n        task_group_up1  task_group_up2\n            (task_up1)  (task_up2)\n                 ╲       ╱\n              task_group_middle\n                (task_middle)\n                  ╱      ╲\n        task_group_down1 task_group_down2\n           (task_down1) (task_down2)\n                 ╲        ╱\n                   ╲    ╱\n                    end\n        '
        execution_date = datetime(2020, 1, 1)
        with DAG(dag_id='test_task_group_sorted', start_date=execution_date) as dag:
            start = EmptyOperator(task_id='start')
            with TaskGroup('task_group_up1') as task_group_up1:
                _ = EmptyOperator(task_id='task_up1')
            with TaskGroup('task_group_up2') as task_group_up2:
                _ = EmptyOperator(task_id='task_up2')
            with TaskGroup('task_group_middle') as task_group_middle:
                _ = EmptyOperator(task_id='task_middle')
            with TaskGroup('task_group_down1') as task_group_down1:
                _ = EmptyOperator(task_id='task_down1')
            with TaskGroup('task_group_down2') as task_group_down2:
                _ = EmptyOperator(task_id='task_down2')
            end = EmptyOperator(task_id='end')
            start >> task_group_up1
            start >> task_group_up2
            task_group_up1 >> task_group_middle
            task_group_up2 >> task_group_middle
            task_group_middle >> task_group_down1
            task_group_middle >> task_group_down2
            task_group_down1 >> end
            task_group_down2 >> end
        task_group_middle_dict = TaskGroupSerialization.serialize_task_group(dag.task_group.children['task_group_middle'])
        upstream_group_ids = task_group_middle_dict['upstream_group_ids']
        assert upstream_group_ids == ['task_group_up1', 'task_group_up2']
        upstream_task_ids = task_group_middle_dict['upstream_task_ids']
        assert upstream_task_ids == ['task_group_up1.task_up1', 'task_group_up2.task_up2']
        downstream_group_ids = task_group_middle_dict['downstream_group_ids']
        assert downstream_group_ids == ['task_group_down1', 'task_group_down2']
        task_group_down1_dict = TaskGroupSerialization.serialize_task_group(dag.task_group.children['task_group_down1'])
        downstream_task_ids = task_group_down1_dict['downstream_task_ids']
        assert downstream_task_ids == ['end']

    def test_edge_info_serialization(self):
        if False:
            return 10
        '\n        Tests edge_info serialization/deserialization.\n        '
        from airflow.operators.empty import EmptyOperator
        from airflow.utils.edgemodifier import Label
        with DAG('test_edge_info_serialization', start_date=datetime(2020, 1, 1)) as dag:
            task1 = EmptyOperator(task_id='task1')
            task2 = EmptyOperator(task_id='task2')
            task1 >> Label('test label') >> task2
        dag_dict = SerializedDAG.to_dict(dag)
        SerializedDAG.validate_schema(dag_dict)
        json_dag = SerializedDAG.from_json(SerializedDAG.to_json(dag))
        self.validate_deserialized_dag(json_dag, dag)
        serialized_dag = SerializedDAG.deserialize_dag(SerializedDAG.serialize_dag(dag))
        assert serialized_dag.edge_info == dag.edge_info

    @pytest.mark.db_test
    @pytest.mark.parametrize('mode', ['poke', 'reschedule'])
    def test_serialize_sensor(self, mode):
        if False:
            while True:
                i = 10
        from airflow.sensors.base import BaseSensorOperator

        class DummySensor(BaseSensorOperator):

            def poke(self, context: Context):
                if False:
                    for i in range(10):
                        print('nop')
                return False
        op = DummySensor(task_id='dummy', mode=mode, poke_interval=23)
        blob = SerializedBaseOperator.serialize_operator(op)
        assert 'deps' in blob
        serialized_op = SerializedBaseOperator.deserialize_operator(blob)
        assert serialized_op.reschedule == (mode == 'reschedule')
        assert op.deps == serialized_op.deps

    @pytest.mark.parametrize('mode', ['poke', 'reschedule'])
    def test_serialize_mapped_sensor_has_reschedule_dep(self, mode):
        if False:
            while True:
                i = 10
        from airflow.sensors.base import BaseSensorOperator

        class DummySensor(BaseSensorOperator):

            def poke(self, context: Context):
                if False:
                    while True:
                        i = 10
                return False
        op = DummySensor.partial(task_id='dummy', mode=mode).expand(poke_interval=[23])
        blob = SerializedBaseOperator.serialize_mapped_operator(op)
        assert 'deps' in blob
        assert 'airflow.ti_deps.deps.ready_to_reschedule.ReadyToRescheduleDep' in blob['deps']

    @pytest.mark.parametrize('passed_success_callback, expected_value', [({'on_success_callback': lambda x: print('hi')}, True), ({}, False)])
    def test_dag_on_success_callback_roundtrip(self, passed_success_callback, expected_value):
        if False:
            print('Hello World!')
        '\n        Test that when on_success_callback is passed to the DAG, has_on_success_callback is stored\n        in Serialized JSON blob. And when it is de-serialized dag.has_on_success_callback is set to True.\n\n        When the callback is not set, has_on_success_callback should not be stored in Serialized blob\n        and so default to False on de-serialization\n        '
        dag = DAG(dag_id='test_dag_on_success_callback_roundtrip', **passed_success_callback)
        BaseOperator(task_id='simple_task', dag=dag, start_date=datetime(2019, 8, 1))
        serialized_dag = SerializedDAG.to_dict(dag)
        if expected_value:
            assert 'has_on_success_callback' in serialized_dag['dag']
        else:
            assert 'has_on_success_callback' not in serialized_dag['dag']
        deserialized_dag = SerializedDAG.from_dict(serialized_dag)
        assert deserialized_dag.has_on_success_callback is expected_value

    @pytest.mark.parametrize('passed_failure_callback, expected_value', [({'on_failure_callback': lambda x: print('hi')}, True), ({}, False)])
    def test_dag_on_failure_callback_roundtrip(self, passed_failure_callback, expected_value):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test that when on_failure_callback is passed to the DAG, has_on_failure_callback is stored\n        in Serialized JSON blob. And when it is de-serialized dag.has_on_failure_callback is set to True.\n\n        When the callback is not set, has_on_failure_callback should not be stored in Serialized blob\n        and so default to False on de-serialization\n        '
        dag = DAG(dag_id='test_dag_on_failure_callback_roundtrip', **passed_failure_callback)
        BaseOperator(task_id='simple_task', dag=dag, start_date=datetime(2019, 8, 1))
        serialized_dag = SerializedDAG.to_dict(dag)
        if expected_value:
            assert 'has_on_failure_callback' in serialized_dag['dag']
        else:
            assert 'has_on_failure_callback' not in serialized_dag['dag']
        deserialized_dag = SerializedDAG.from_dict(serialized_dag)
        assert deserialized_dag.has_on_failure_callback is expected_value

    @pytest.mark.parametrize('object_to_serialized, expected_output', [(['task_1', 'task_5', 'task_2', 'task_4'], ['task_1', 'task_5', 'task_2', 'task_4']), ({'task_1', 'task_5', 'task_2', 'task_4'}, ['task_1', 'task_2', 'task_4', 'task_5']), (('task_1', 'task_5', 'task_2', 'task_4'), ['task_1', 'task_5', 'task_2', 'task_4']), ({'staging_schema': [{'key:': 'foo', 'value': 'bar'}, {'key:': 'this', 'value': 'that'}, 'test_conf']}, {'staging_schema': [{'__type': 'dict', '__var': {'key:': 'foo', 'value': 'bar'}}, {'__type': 'dict', '__var': {'key:': 'this', 'value': 'that'}}, 'test_conf']}), ({'task3': 'test3', 'task2': 'test2', 'task1': 'test1'}, {'task1': 'test1', 'task2': 'test2', 'task3': 'test3'}), (('task_1', 'task_5', 'task_2', 3, ['x', 'y']), ['task_1', 'task_5', 'task_2', 3, ['x', 'y']])])
    def test_serialized_objects_are_sorted(self, object_to_serialized, expected_output):
        if False:
            i = 10
            return i + 15
        'Test Serialized Sets are sorted while list and tuple preserve order'
        serialized_obj = SerializedDAG.serialize(object_to_serialized)
        if isinstance(serialized_obj, dict) and '__type' in serialized_obj:
            serialized_obj = serialized_obj['__var']
        assert serialized_obj == expected_output

    def test_params_upgrade(self):
        if False:
            while True:
                i = 10
        'when pre-2.2.0 param (i.e. primitive) is deserialized we convert to Param'
        serialized = {'__version': 1, 'dag': {'_dag_id': 'simple_dag', 'fileloc': '/path/to/file.py', 'tasks': [], 'timezone': 'UTC', 'params': {'none': None, 'str': 'str', 'dict': {'a': 'b'}}}}
        dag = SerializedDAG.from_dict(serialized)
        assert dag.params['none'] is None
        assert isinstance(dag.params.get_param('none'), Param)
        assert dag.params['str'] == 'str'

    def test_params_serialize_default_2_2_0(self):
        if False:
            return 10
        'In 2.0.0, param ``default`` was assumed to be json-serializable objects and were not run though\n        the standard serializer function.  In 2.2.2 we serialize param ``default``.  We keep this\n        test only to ensure that params stored in 2.2.0 can still be parsed correctly.'
        serialized = {'__version': 1, 'dag': {'_dag_id': 'simple_dag', 'fileloc': '/path/to/file.py', 'tasks': [], 'timezone': 'UTC', 'params': {'str': {'__class': 'airflow.models.param.Param', 'default': 'str'}}}}
        SerializedDAG.validate_schema(serialized)
        dag = SerializedDAG.from_dict(serialized)
        assert isinstance(dag.params.get_param('str'), Param)
        assert dag.params['str'] == 'str'

    def test_params_serialize_default(self):
        if False:
            i = 10
            return i + 15
        serialized = {'__version': 1, 'dag': {'_dag_id': 'simple_dag', 'fileloc': '/path/to/file.py', 'tasks': [], 'timezone': 'UTC', 'params': {'my_param': {'default': 'a string value', 'description': 'hello', 'schema': {'__var': {'type': 'string'}, '__type': 'dict'}, '__class': 'airflow.models.param.Param'}}}}
        SerializedDAG.validate_schema(serialized)
        dag = SerializedDAG.from_dict(serialized)
        assert dag.params['my_param'] == 'a string value'
        param = dag.params.get_param('my_param')
        assert isinstance(param, Param)
        assert param.description == 'hello'
        assert param.schema == {'type': 'string'}

    @pytest.mark.db_test
    def test_not_templateable_fields_in_serialized_dag(self):
        if False:
            print('Hello World!')
        '\n        Test that when we use not templateable fields, an Airflow exception is raised.\n        '

        class TestOperator(BaseOperator):
            template_fields = ('email', 'execution_timeout')

            def execute(self, context: Context):
                if False:
                    return 10
                pass
        dag = DAG(dag_id='test_dag', start_date=datetime(2023, 11, 9))
        with dag:
            task = TestOperator(task_id='test_task', email="{{ ','.join(test_email_list) }}", execution_timeout=timedelta(seconds=10))
            task.render_template_fields(context={'test_email_list': ['foo@test.com', 'bar@test.com']})
            assert task.email == 'foo@test.com,bar@test.com'
        with pytest.raises(AirflowException, match="Cannot template BaseOperator field: 'execution_timeout'"):
            SerializedDAG.to_dict(dag)

def test_kubernetes_optional():
    if False:
        print('Hello World!')
    'Serialisation / deserialisation continues to work without kubernetes installed'

    def mock__import__(name, globals_=None, locals_=None, fromlist=(), level=0):
        if False:
            i = 10
            return i + 15
        if level == 0 and name.partition('.')[0] == 'kubernetes':
            raise ImportError("No module named 'kubernetes'")
        return importlib.__import__(name, globals=globals_, locals=locals_, fromlist=fromlist, level=level)
    with mock.patch('builtins.__import__', side_effect=mock__import__) as import_mock:
        spec = importlib.util.find_spec('airflow.serialization.serialized_objects')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        imported_airflow = {c.args[0].split('.', 2)[1] for c in import_mock.call_args_list if c.args[0].startswith('airflow.')}
        assert 'kubernetes' not in imported_airflow
        pod_override = {'__type': 'k8s.V1Pod', '__var': PodGenerator.serialize_pod(executor_config_pod)}
        with pytest.raises(RuntimeError):
            module.BaseSerialization.from_dict(pod_override)
        module.SerializedDAG.to_dict(make_simple_dag()['simple_dag'])

def test_operator_expand_serde():
    if False:
        for i in range(10):
            print('nop')
    literal = [1, 2, {'a': 'b'}]
    real_op = BashOperator.partial(task_id='a', executor_config={'dict': {'sub': 'value'}}).expand(bash_command=literal)
    serialized = SerializedBaseOperator.serialize(real_op)
    assert serialized == {'_is_empty': False, '_is_mapped': True, '_task_module': 'airflow.operators.bash', '_task_type': 'BashOperator', 'downstream_task_ids': [], 'expand_input': {'type': 'dict-of-lists', 'value': {'__type': 'dict', '__var': {'bash_command': [1, 2, {'__type': 'dict', '__var': {'a': 'b'}}]}}}, 'partial_kwargs': {'executor_config': {'__type': 'dict', '__var': {'dict': {'__type': 'dict', '__var': {'sub': 'value'}}}}}, 'task_id': 'a', 'operator_extra_links': [], 'template_fields': ['bash_command', 'env'], 'template_ext': ['.sh', '.bash'], 'template_fields_renderers': {'bash_command': 'bash', 'env': 'json'}, 'ui_color': '#f0ede4', 'ui_fgcolor': '#000', '_disallow_kwargs_override': False, '_expand_input_attr': 'expand_input'}
    op = SerializedBaseOperator.deserialize_operator(serialized)
    assert isinstance(op, MappedOperator)
    assert op.deps is MappedOperator.deps_for(BaseOperator)
    assert op.operator_class == {'_task_type': 'BashOperator', 'downstream_task_ids': [], 'task_id': 'a', 'template_ext': ['.sh', '.bash'], 'template_fields': ['bash_command', 'env'], 'template_fields_renderers': {'bash_command': 'bash', 'env': 'json'}, 'ui_color': '#f0ede4', 'ui_fgcolor': '#000'}
    assert op.expand_input.value['bash_command'] == literal
    assert op.partial_kwargs['executor_config'] == {'dict': {'sub': 'value'}}

def test_operator_expand_xcomarg_serde():
    if False:
        while True:
            i = 10
    from airflow.models.xcom_arg import PlainXComArg, XComArg
    from airflow.serialization.serialized_objects import _XComRef
    with DAG('test-dag', start_date=datetime(2020, 1, 1)) as dag:
        task1 = BaseOperator(task_id='op1')
        mapped = MockOperator.partial(task_id='task_2').expand(arg2=XComArg(task1))
    serialized = SerializedBaseOperator.serialize(mapped)
    assert serialized == {'_is_empty': False, '_is_mapped': True, '_task_module': 'tests.test_utils.mock_operators', '_task_type': 'MockOperator', 'downstream_task_ids': [], 'expand_input': {'type': 'dict-of-lists', 'value': {'__type': 'dict', '__var': {'arg2': {'__type': 'xcomref', '__var': {'task_id': 'op1', 'key': 'return_value'}}}}}, 'partial_kwargs': {}, 'task_id': 'task_2', 'template_fields': ['arg1', 'arg2'], 'template_ext': [], 'template_fields_renderers': {}, 'operator_extra_links': [], 'ui_color': '#fff', 'ui_fgcolor': '#000', '_disallow_kwargs_override': False, '_expand_input_attr': 'expand_input'}
    op = SerializedBaseOperator.deserialize_operator(serialized)
    assert op.deps is MappedOperator.deps_for(BaseOperator)
    xcom_ref = op.expand_input.value['arg2']
    assert xcom_ref == _XComRef({'task_id': 'op1', 'key': XCOM_RETURN_KEY})
    serialized_dag: DAG = SerializedDAG.from_dict(SerializedDAG.to_dict(dag))
    xcom_arg = serialized_dag.task_dict['task_2'].expand_input.value['arg2']
    assert isinstance(xcom_arg, PlainXComArg)
    assert xcom_arg.operator is serialized_dag.task_dict['op1']

@pytest.mark.parametrize('strict', [True, False])
def test_operator_expand_kwargs_literal_serde(strict):
    if False:
        i = 10
        return i + 15
    from airflow.models.xcom_arg import PlainXComArg, XComArg
    from airflow.serialization.serialized_objects import _XComRef
    with DAG('test-dag', start_date=datetime(2020, 1, 1)) as dag:
        task1 = BaseOperator(task_id='op1')
        mapped = MockOperator.partial(task_id='task_2').expand_kwargs([{'a': 'x'}, {'a': XComArg(task1)}], strict=strict)
    serialized = SerializedBaseOperator.serialize(mapped)
    assert serialized == {'_is_empty': False, '_is_mapped': True, '_task_module': 'tests.test_utils.mock_operators', '_task_type': 'MockOperator', 'downstream_task_ids': [], 'expand_input': {'type': 'list-of-dicts', 'value': [{'__type': 'dict', '__var': {'a': 'x'}}, {'__type': 'dict', '__var': {'a': {'__type': 'xcomref', '__var': {'task_id': 'op1', 'key': 'return_value'}}}}]}, 'partial_kwargs': {}, 'task_id': 'task_2', 'template_fields': ['arg1', 'arg2'], 'template_ext': [], 'template_fields_renderers': {}, 'operator_extra_links': [], 'ui_color': '#fff', 'ui_fgcolor': '#000', '_disallow_kwargs_override': strict, '_expand_input_attr': 'expand_input'}
    op = SerializedBaseOperator.deserialize_operator(serialized)
    assert op.deps is MappedOperator.deps_for(BaseOperator)
    assert op._disallow_kwargs_override == strict
    expand_value = op.expand_input.value
    assert expand_value == [{'a': 'x'}, {'a': _XComRef({'task_id': 'op1', 'key': XCOM_RETURN_KEY})}]
    serialized_dag: DAG = SerializedDAG.from_dict(SerializedDAG.to_dict(dag))
    resolved_expand_value = serialized_dag.task_dict['task_2'].expand_input.value
    resolved_expand_value == [{'a': 'x'}, {'a': PlainXComArg(serialized_dag.task_dict['op1'])}]

@pytest.mark.parametrize('strict', [True, False])
def test_operator_expand_kwargs_xcomarg_serde(strict):
    if False:
        i = 10
        return i + 15
    from airflow.models.xcom_arg import PlainXComArg, XComArg
    from airflow.serialization.serialized_objects import _XComRef
    with DAG('test-dag', start_date=datetime(2020, 1, 1)) as dag:
        task1 = BaseOperator(task_id='op1')
        mapped = MockOperator.partial(task_id='task_2').expand_kwargs(XComArg(task1), strict=strict)
    serialized = SerializedBaseOperator.serialize(mapped)
    assert serialized == {'_is_empty': False, '_is_mapped': True, '_task_module': 'tests.test_utils.mock_operators', '_task_type': 'MockOperator', 'downstream_task_ids': [], 'expand_input': {'type': 'list-of-dicts', 'value': {'__type': 'xcomref', '__var': {'task_id': 'op1', 'key': 'return_value'}}}, 'partial_kwargs': {}, 'task_id': 'task_2', 'template_fields': ['arg1', 'arg2'], 'template_ext': [], 'template_fields_renderers': {}, 'operator_extra_links': [], 'ui_color': '#fff', 'ui_fgcolor': '#000', '_disallow_kwargs_override': strict, '_expand_input_attr': 'expand_input'}
    op = SerializedBaseOperator.deserialize_operator(serialized)
    assert op.deps is MappedOperator.deps_for(BaseOperator)
    assert op._disallow_kwargs_override == strict
    xcom_ref = op.expand_input.value
    assert xcom_ref == _XComRef({'task_id': 'op1', 'key': XCOM_RETURN_KEY})
    serialized_dag: DAG = SerializedDAG.from_dict(SerializedDAG.to_dict(dag))
    xcom_arg = serialized_dag.task_dict['task_2'].expand_input.value
    assert isinstance(xcom_arg, PlainXComArg)
    assert xcom_arg.operator is serialized_dag.task_dict['op1']

def test_operator_expand_deserialized_unmap():
    if False:
        while True:
            i = 10
    'Unmap a deserialized mapped operator should be similar to deserializing an non-mapped operator.'
    normal = BashOperator(task_id='a', bash_command=[1, 2], executor_config={'a': 'b'})
    mapped = BashOperator.partial(task_id='a', executor_config={'a': 'b'}).expand(bash_command=[1, 2])
    serialize = SerializedBaseOperator.serialize
    deserialize = SerializedBaseOperator.deserialize_operator
    assert deserialize(serialize(mapped)).unmap(None) == deserialize(serialize(normal))

@pytest.mark.db_test
def test_sensor_expand_deserialized_unmap():
    if False:
        for i in range(10):
            print('nop')
    'Unmap a deserialized mapped sensor should be similar to deserializing a non-mapped sensor'
    normal = BashSensor(task_id='a', bash_command=[1, 2], mode='reschedule')
    mapped = BashSensor.partial(task_id='a', mode='reschedule').expand(bash_command=[1, 2])
    serialize = SerializedBaseOperator.serialize
    deserialize = SerializedBaseOperator.deserialize_operator
    assert deserialize(serialize(mapped)).unmap(None) == deserialize(serialize(normal))

def test_task_resources_serde():
    if False:
        while True:
            i = 10
    '\n    Test task resources serialization/deserialization.\n    '
    from airflow.operators.empty import EmptyOperator
    execution_date = datetime(2020, 1, 1)
    task_id = 'task1'
    with DAG('test_task_resources', start_date=execution_date) as _:
        task = EmptyOperator(task_id=task_id, resources={'cpus': 0.1, 'ram': 2048})
    serialized = SerializedBaseOperator.serialize(task)
    assert serialized['resources'] == {'cpus': {'name': 'CPU', 'qty': 0.1, 'units_str': 'core(s)'}, 'disk': {'name': 'Disk', 'qty': 512, 'units_str': 'MB'}, 'gpus': {'name': 'GPU', 'qty': 0, 'units_str': 'gpu(s)'}, 'ram': {'name': 'RAM', 'qty': 2048, 'units_str': 'MB'}}

def test_taskflow_expand_serde():
    if False:
        print('Hello World!')
    from airflow.decorators import task
    from airflow.models.xcom_arg import XComArg
    from airflow.serialization.serialized_objects import _ExpandInputRef, _XComRef
    with DAG('test-dag', start_date=datetime(2020, 1, 1)) as dag:
        op1 = BaseOperator(task_id='op1')

        @task(retry_delay=30)
        def x(arg1, arg2, arg3):
            if False:
                return 10
            print(arg1, arg2, arg3)
        print('**', type(x), type(x.partial), type(x.expand))
        x.partial(arg1=[1, 2, {'a': 'b'}]).expand(arg2={'a': 1, 'b': 2}, arg3=XComArg(op1))
    original = dag.get_task('x')
    serialized = SerializedBaseOperator.serialize(original)
    assert serialized == {'_is_empty': False, '_is_mapped': True, '_task_module': 'airflow.decorators.python', '_task_type': '_PythonDecoratedOperator', '_operator_name': '@task', 'downstream_task_ids': [], 'partial_kwargs': {'is_setup': False, 'is_teardown': False, 'on_failure_fail_dagrun': False, 'op_args': [], 'op_kwargs': {'__type': 'dict', '__var': {'arg1': [1, 2, {'__type': 'dict', '__var': {'a': 'b'}}]}}, 'retry_delay': {'__type': 'timedelta', '__var': 30.0}}, 'op_kwargs_expand_input': {'type': 'dict-of-lists', 'value': {'__type': 'dict', '__var': {'arg2': {'__type': 'dict', '__var': {'a': 1, 'b': 2}}, 'arg3': {'__type': 'xcomref', '__var': {'task_id': 'op1', 'key': 'return_value'}}}}}, 'operator_extra_links': [], 'ui_color': '#ffefeb', 'ui_fgcolor': '#000', 'task_id': 'x', 'template_ext': [], 'template_fields': ['templates_dict', 'op_args', 'op_kwargs'], 'template_fields_renderers': {'templates_dict': 'json', 'op_args': 'py', 'op_kwargs': 'py'}, '_disallow_kwargs_override': False, '_expand_input_attr': 'op_kwargs_expand_input'}
    deserialized = SerializedBaseOperator.deserialize_operator(serialized)
    assert isinstance(deserialized, MappedOperator)
    assert deserialized.deps is MappedOperator.deps_for(BaseOperator)
    assert deserialized.upstream_task_ids == set()
    assert deserialized.downstream_task_ids == set()
    assert deserialized.op_kwargs_expand_input == _ExpandInputRef(key='dict-of-lists', value={'arg2': {'a': 1, 'b': 2}, 'arg3': _XComRef({'task_id': 'op1', 'key': XCOM_RETURN_KEY})})
    assert deserialized.partial_kwargs == {'is_setup': False, 'is_teardown': False, 'on_failure_fail_dagrun': False, 'op_args': [], 'op_kwargs': {'arg1': [1, 2, {'a': 'b'}]}, 'retry_delay': timedelta(seconds=30)}
    pickled = pickle.loads(pickle.dumps(deserialized))
    assert pickled.op_kwargs_expand_input == _ExpandInputRef(key='dict-of-lists', value={'arg2': {'a': 1, 'b': 2}, 'arg3': _XComRef({'task_id': 'op1', 'key': XCOM_RETURN_KEY})})
    assert pickled.partial_kwargs == {'is_setup': False, 'is_teardown': False, 'on_failure_fail_dagrun': False, 'op_args': [], 'op_kwargs': {'arg1': [1, 2, {'a': 'b'}]}, 'retry_delay': timedelta(seconds=30)}

@pytest.mark.parametrize('strict', [True, False])
def test_taskflow_expand_kwargs_serde(strict):
    if False:
        i = 10
        return i + 15
    from airflow.decorators import task
    from airflow.models.xcom_arg import XComArg
    from airflow.serialization.serialized_objects import _ExpandInputRef, _XComRef
    with DAG('test-dag', start_date=datetime(2020, 1, 1)) as dag:
        op1 = BaseOperator(task_id='op1')

        @task(retry_delay=30)
        def x(arg1, arg2, arg3):
            if False:
                for i in range(10):
                    print('nop')
            print(arg1, arg2, arg3)
        x.partial(arg1=[1, 2, {'a': 'b'}]).expand_kwargs(XComArg(op1), strict=strict)
    original = dag.get_task('x')
    serialized = SerializedBaseOperator.serialize(original)
    assert serialized == {'_is_empty': False, '_is_mapped': True, '_task_module': 'airflow.decorators.python', '_task_type': '_PythonDecoratedOperator', '_operator_name': '@task', 'downstream_task_ids': [], 'partial_kwargs': {'is_setup': False, 'is_teardown': False, 'on_failure_fail_dagrun': False, 'op_args': [], 'op_kwargs': {'__type': 'dict', '__var': {'arg1': [1, 2, {'__type': 'dict', '__var': {'a': 'b'}}]}}, 'retry_delay': {'__type': 'timedelta', '__var': 30.0}}, 'op_kwargs_expand_input': {'type': 'list-of-dicts', 'value': {'__type': 'xcomref', '__var': {'task_id': 'op1', 'key': 'return_value'}}}, 'operator_extra_links': [], 'ui_color': '#ffefeb', 'ui_fgcolor': '#000', 'task_id': 'x', 'template_ext': [], 'template_fields': ['templates_dict', 'op_args', 'op_kwargs'], 'template_fields_renderers': {'templates_dict': 'json', 'op_args': 'py', 'op_kwargs': 'py'}, '_disallow_kwargs_override': strict, '_expand_input_attr': 'op_kwargs_expand_input'}
    deserialized = SerializedBaseOperator.deserialize_operator(serialized)
    assert isinstance(deserialized, MappedOperator)
    assert deserialized.deps is MappedOperator.deps_for(BaseOperator)
    assert deserialized._disallow_kwargs_override == strict
    assert deserialized.upstream_task_ids == set()
    assert deserialized.downstream_task_ids == set()
    assert deserialized.op_kwargs_expand_input == _ExpandInputRef(key='list-of-dicts', value=_XComRef({'task_id': 'op1', 'key': XCOM_RETURN_KEY}))
    assert deserialized.partial_kwargs == {'is_setup': False, 'is_teardown': False, 'on_failure_fail_dagrun': False, 'op_args': [], 'op_kwargs': {'arg1': [1, 2, {'a': 'b'}]}, 'retry_delay': timedelta(seconds=30)}
    pickled = pickle.loads(pickle.dumps(deserialized))
    assert pickled.op_kwargs_expand_input == _ExpandInputRef('list-of-dicts', _XComRef({'task_id': 'op1', 'key': XCOM_RETURN_KEY}))
    assert pickled.partial_kwargs == {'is_setup': False, 'is_teardown': False, 'on_failure_fail_dagrun': False, 'op_args': [], 'op_kwargs': {'arg1': [1, 2, {'a': 'b'}]}, 'retry_delay': timedelta(seconds=30)}

def test_mapped_task_group_serde():
    if False:
        for i in range(10):
            print('nop')
    from airflow.decorators.task_group import task_group
    from airflow.models.expandinput import DictOfListsExpandInput
    from airflow.utils.task_group import MappedTaskGroup
    with DAG('test-dag', start_date=datetime(2020, 1, 1)) as dag:

        @task_group
        def tg(a: str) -> None:
            if False:
                i = 10
                return i + 15
            BaseOperator(task_id='op1')
            with pytest.raises(NotImplementedError) as ctx:
                BashOperator.partial(task_id='op2').expand(bash_command=['ls', a])
            assert str(ctx.value) == 'operator expansion in an expanded task group is not yet supported'
        tg.expand(a=['.', '..'])
    ser_dag = SerializedBaseOperator.serialize(dag)
    assert ser_dag[Encoding.VAR]['_task_group']['children']['tg'] == ('taskgroup', {'_group_id': 'tg', 'children': {'tg.op1': ('operator', 'tg.op1')}, 'downstream_group_ids': [], 'downstream_task_ids': [], 'expand_input': {'type': 'dict-of-lists', 'value': {'__type': 'dict', '__var': {'a': ['.', '..']}}}, 'is_mapped': True, 'prefix_group_id': True, 'tooltip': '', 'ui_color': 'CornflowerBlue', 'ui_fgcolor': '#000', 'upstream_group_ids': [], 'upstream_task_ids': []})
    serde_dag = SerializedDAG.deserialize_dag(ser_dag[Encoding.VAR])
    serde_tg = serde_dag.task_group.children['tg']
    assert isinstance(serde_tg, MappedTaskGroup)
    assert serde_tg._expand_input == DictOfListsExpandInput({'a': ['.', '..']})

@pytest.mark.db_test
def test_mapped_task_with_operator_extra_links_property():
    if False:
        i = 10
        return i + 15

    class _DummyOperator(BaseOperator):

        def __init__(self, inputs, **kwargs):
            if False:
                print('Hello World!')
            super().__init__(**kwargs)
            self.inputs = inputs

        @property
        def operator_extra_links(self):
            if False:
                print('Hello World!')
            return (AirflowLink2(),)
    with DAG('test-dag', start_date=datetime(2020, 1, 1)) as dag:
        _DummyOperator.partial(task_id='task').expand(inputs=[1, 2, 3])
    serialized_dag = SerializedBaseOperator.serialize(dag)
    assert serialized_dag[Encoding.VAR]['tasks'][0] == {'task_id': 'task', 'expand_input': {'type': 'dict-of-lists', 'value': {'__type': 'dict', '__var': {'inputs': [1, 2, 3]}}}, 'partial_kwargs': {}, '_disallow_kwargs_override': False, '_expand_input_attr': 'expand_input', 'downstream_task_ids': [], '_operator_extra_links': [{'tests.test_utils.mock_operators.AirflowLink2': {}}], 'ui_color': '#fff', 'ui_fgcolor': '#000', 'template_ext': [], 'template_fields': [], 'template_fields_renderers': {}, '_task_type': '_DummyOperator', '_task_module': 'tests.serialization.test_dag_serialization', '_is_empty': False, '_is_mapped': True}
    deserialized_dag = SerializedDAG.deserialize_dag(serialized_dag[Encoding.VAR])
    assert deserialized_dag.task_dict['task'].operator_extra_links == [AirflowLink2()]