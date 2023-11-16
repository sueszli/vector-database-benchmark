import hashlib
import json
import sys
import platform
from collections import defaultdict
from datetime import datetime, timedelta
TASK_ID_XCOM_KEY = 'metaflow_task_id'
FOREACH_CARDINALITY_XCOM_KEY = 'metaflow_foreach_cardinality'
FOREACH_XCOM_KEY = 'metaflow_foreach_indexes'
RUN_HASH_ID_LEN = 12
TASK_ID_HASH_LEN = 8
RUN_ID_PREFIX = 'airflow'
AIRFLOW_FOREACH_SUPPORT_VERSION = '2.3.0'
AIRFLOW_MIN_SUPPORT_VERSION = '2.2.0'
KUBERNETES_PROVIDER_FOREACH_VERSION = '4.2.0'

class KubernetesProviderNotFound(Exception):
    headline = 'Kubernetes provider not found'

class ForeachIncompatibleException(Exception):
    headline = 'Airflow version is incompatible to support Metaflow `foreach`s.'

class IncompatibleVersionException(Exception):
    headline = 'Metaflow is incompatible with current version of Airflow.'

    def __init__(self, version_number) -> None:
        if False:
            for i in range(10):
                print('nop')
        msg = 'Airflow version %s is incompatible with Metaflow. Metaflow requires Airflow a minimum version %s' % (version_number, AIRFLOW_MIN_SUPPORT_VERSION)
        super().__init__(msg)

class IncompatibleKubernetesProviderVersionException(Exception):
    headline = 'Kubernetes Provider version is incompatible with Metaflow `foreach`s. Install the provider via `%s -m pip install apache-airflow-providers-cncf-kubernetes==%s`' % (sys.executable, KUBERNETES_PROVIDER_FOREACH_VERSION)

class AirflowSensorNotFound(Exception):
    headline = 'Sensor package not found'

def create_absolute_version_number(version):
    if False:
        while True:
            i = 10
    abs_version = None
    if all((v.isdigit() for v in version.split('.'))):
        abs_version = sum([10 ** (3 - idx) * i for (idx, i) in enumerate([int(v) for v in version.split('.')])])
    elif all((v.isdigit() for v in version.split('.')[:2])):
        abs_version = sum([10 ** (3 - idx) * i for (idx, i) in enumerate([int(v) for v in version.split('.')[:2]])])
    return abs_version

def _validate_dynamic_mapping_compatibility():
    if False:
        i = 10
        return i + 15
    from airflow.version import version
    af_ver = create_absolute_version_number(version)
    if af_ver is None or af_ver < create_absolute_version_number(AIRFLOW_FOREACH_SUPPORT_VERSION):
        ForeachIncompatibleException("Please install airflow version %s to use Airflow's Dynamic task mapping functionality." % AIRFLOW_FOREACH_SUPPORT_VERSION)

def get_kubernetes_provider_version():
    if False:
        for i in range(10):
            print('nop')
    try:
        from airflow.providers.cncf.kubernetes.get_provider_info import get_provider_info
    except ImportError as e:
        raise KubernetesProviderNotFound('This DAG utilizes `KubernetesPodOperator`. Install the Airflow Kubernetes provider using `%s -m pip install apache-airflow-providers-cncf-kubernetes`' % sys.executable)
    return get_provider_info()['versions'][0]

def _validate_minimum_airflow_version():
    if False:
        return 10
    from airflow.version import version
    af_ver = create_absolute_version_number(version)
    if af_ver is None or af_ver < create_absolute_version_number(AIRFLOW_MIN_SUPPORT_VERSION):
        raise IncompatibleVersionException(version)

def _check_foreach_compatible_kubernetes_provider():
    if False:
        while True:
            i = 10
    provider_version = get_kubernetes_provider_version()
    ver = create_absolute_version_number(provider_version)
    if ver is None or ver < create_absolute_version_number(KUBERNETES_PROVIDER_FOREACH_VERSION):
        raise IncompatibleKubernetesProviderVersionException()

def datetimeparse(isotimestamp):
    if False:
        print('Hello World!')
    ver = int(platform.python_version_tuple()[0]) * 10 + int(platform.python_version_tuple()[1])
    if ver >= 37:
        return datetime.fromisoformat(isotimestamp)
    else:
        return datetime.strptime(isotimestamp, '%Y-%m-%dT%H:%M:%S.%f')

def get_xcom_arg_class():
    if False:
        while True:
            i = 10
    try:
        from airflow import XComArg
    except ImportError:
        return None
    return XComArg

class AIRFLOW_MACROS:
    RUN_ID = '%s-{{ [run_id, dag_run.dag_id] | run_id_creator }}' % RUN_ID_PREFIX
    PARAMETERS = '{{ params | json_dump }}'
    STEPNAME = '{{ ti.task_id }}'
    TASK_ID = '%s-{{ [run_id, ti.task_id, dag_run.dag_id] | task_id_creator  }}' % RUN_ID_PREFIX
    FOREACH_TASK_ID = '%s-{{ [run_id, ti.task_id, dag_run.dag_id, ti.map_index] | task_id_creator  }}' % RUN_ID_PREFIX
    RUN_ID_SHELL = "%s-$(echo -n {{ run_id }}-{{ dag_run.dag_id }} | md5sum | awk '{print $1}' | awk '{print substr ($0, 0, %s)}')" % (RUN_ID_PREFIX, str(RUN_HASH_ID_LEN))
    ATTEMPT = '{{ task_instance.try_number - 1 }}'
    AIRFLOW_RUN_ID = '{{ run_id }}'
    AIRFLOW_JOB_ID = '{{ ti.job_id }}'
    FOREACH_SPLIT_INDEX = '{{ ti.map_index }}'

    @classmethod
    def create_task_id(cls, is_foreach):
        if False:
            for i in range(10):
                print('nop')
        if is_foreach:
            return cls.FOREACH_TASK_ID
        else:
            return cls.TASK_ID

    @classmethod
    def pathspec(cls, flowname, is_foreach=False):
        if False:
            while True:
                i = 10
        return '%s/%s/%s/%s' % (flowname, cls.RUN_ID, cls.STEPNAME, cls.create_task_id(is_foreach))

class SensorNames:
    EXTERNAL_TASK_SENSOR = 'ExternalTaskSensor'
    S3_SENSOR = 'S3KeySensor'

    @classmethod
    def get_supported_sensors(cls):
        if False:
            print('Hello World!')
        return list(cls.__dict__.values())

def run_id_creator(val):
    if False:
        i = 10
        return i + 15
    return hashlib.md5('-'.join([str(x) for x in val]).encode('utf-8')).hexdigest()[:RUN_HASH_ID_LEN]

def task_id_creator(val):
    if False:
        return 10
    return hashlib.md5('-'.join([str(x) for x in val]).encode('utf-8')).hexdigest()[:TASK_ID_HASH_LEN]

def id_creator(val, hash_len):
    if False:
        print('Hello World!')
    return hashlib.md5('-'.join([str(x) for x in val]).encode('utf-8')).hexdigest()[:hash_len]

def json_dump(val):
    if False:
        while True:
            i = 10
    return json.dumps(val)

class AirflowDAGArgs(object):
    _arg_types = {'dag_id': str, 'description': str, 'schedule_interval': str, 'start_date': datetime, 'catchup': bool, 'tags': list, 'dagrun_timeout': timedelta, 'default_args': {'owner': str, 'depends_on_past': bool, 'email': list, 'email_on_failure': bool, 'email_on_retry': bool, 'retries': int, 'retry_delay': timedelta, 'queue': str, 'pool': str, 'priority_weight': int, 'wait_for_downstream': bool, 'sla': timedelta, 'execution_timeout': timedelta, 'trigger_rule': str}}
    filters = dict(task_id_creator=lambda v: task_id_creator(v), json_dump=lambda val: json_dump(val), run_id_creator=lambda val: run_id_creator(val), join_list=lambda x: ','.join(list(x)))

    def __init__(self, **kwargs):
        if False:
            while True:
                i = 10
        self._args = kwargs

    @property
    def arguments(self):
        if False:
            print('Hello World!')
        return dict(**self._args, user_defined_filters=self.filters)

    def serialize(self):
        if False:
            for i in range(10):
                print('nop')

        def parse_args(dd):
            if False:
                for i in range(10):
                    print('nop')
            data_dict = {}
            for (k, v) in dd.items():
                if isinstance(v, dict):
                    data_dict[k] = parse_args(v)
                elif isinstance(v, datetime):
                    data_dict[k] = v.isoformat()
                elif isinstance(v, timedelta):
                    data_dict[k] = dict(seconds=v.total_seconds())
                else:
                    data_dict[k] = v
            return data_dict
        return parse_args(self._args)

    @classmethod
    def deserialize(cls, data_dict):
        if False:
            return 10

        def parse_args(dd, type_check_dict):
            if False:
                print('Hello World!')
            kwrgs = {}
            for (k, v) in dd.items():
                if k not in type_check_dict:
                    kwrgs[k] = v
                elif isinstance(v, dict) and isinstance(type_check_dict[k], dict):
                    kwrgs[k] = parse_args(v, type_check_dict[k])
                elif type_check_dict[k] == datetime:
                    kwrgs[k] = datetimeparse(v)
                elif type_check_dict[k] == timedelta:
                    kwrgs[k] = timedelta(**v)
                else:
                    kwrgs[k] = v
            return kwrgs
        return cls(**parse_args(data_dict, cls._arg_types))

def _kubernetes_pod_operator_args(operator_args):
    if False:
        return 10
    from kubernetes import client
    from airflow.kubernetes.secret import Secret
    secrets = [Secret('env', secret, secret) for secret in operator_args.get('secrets', [])]
    args = operator_args
    args.update({'secrets': secrets})
    additional_env_vars = [client.V1EnvVar(name=k, value_from=client.V1EnvVarSource(field_ref=client.V1ObjectFieldSelector(field_path=str(v)))) for (k, v) in {'METAFLOW_KUBERNETES_POD_NAMESPACE': 'metadata.namespace', 'METAFLOW_KUBERNETES_POD_NAME': 'metadata.name', 'METAFLOW_KUBERNETES_POD_ID': 'metadata.uid', 'METAFLOW_KUBERNETES_SERVICE_ACCOUNT_NAME': 'spec.serviceAccountName', 'METAFLOW_KUBERNETES_NODE_IP': 'status.hostIP'}.items()]
    args['pod_runtime_info_envs'] = additional_env_vars
    resources = args.get('resources')
    provider_version = get_kubernetes_provider_version()
    k8s_op_ver = create_absolute_version_number(provider_version)
    if k8s_op_ver is None or k8s_op_ver < create_absolute_version_number(KUBERNETES_PROVIDER_FOREACH_VERSION):
        args['resources'] = client.V1ResourceRequirements(requests=resources['requests'], limits=None if 'limits' not in resources else resources['limits'])
    else:
        args['container_resources'] = client.V1ResourceRequirements(requests=resources['requests'], limits=None if 'limits' not in resources else resources['limits'])
        del args['resources']
    if operator_args.get('execution_timeout'):
        args['execution_timeout'] = timedelta(**operator_args.get('execution_timeout'))
    if operator_args.get('retry_delay'):
        args['retry_delay'] = timedelta(**operator_args.get('retry_delay'))
    return args

def _parse_sensor_args(name, kwargs):
    if False:
        return 10
    if name == SensorNames.EXTERNAL_TASK_SENSOR:
        if 'execution_delta' in kwargs:
            if type(kwargs['execution_delta']) == dict:
                kwargs['execution_delta'] = timedelta(**kwargs['execution_delta'])
            else:
                del kwargs['execution_delta']
    return kwargs

def _get_sensor(name):
    if False:
        while True:
            i = 10
    if name == SensorNames.EXTERNAL_TASK_SENSOR:
        from airflow.sensors.external_task_sensor import ExternalTaskSensor
        return ExternalTaskSensor
    elif name == SensorNames.S3_SENSOR:
        try:
            from airflow.providers.amazon.aws.sensors.s3 import S3KeySensor
        except ImportError:
            raise AirflowSensorNotFound('This DAG requires a `S3KeySensor`. Install the Airflow AWS provider using : `pip install apache-airflow-providers-amazon`')
        return S3KeySensor

def get_metaflow_kubernetes_operator():
    if False:
        print('Hello World!')
    try:
        from airflow.contrib.operators.kubernetes_pod_operator import KubernetesPodOperator
    except ImportError:
        try:
            from airflow.providers.cncf.kubernetes.operators.kubernetes_pod import KubernetesPodOperator
        except ImportError as e:
            raise KubernetesProviderNotFound('This DAG utilizes `KubernetesPodOperator`. Install the Airflow Kubernetes provider using `%s -m pip install apache-airflow-providers-cncf-kubernetes`' % sys.executable)

    class MetaflowKubernetesOperator(KubernetesPodOperator):
        """
        ## Why Inherit the `KubernetesPodOperator` class ?

        Two key reasons :

        1. So that we can override the `execute` method.
        The only change we introduce to the method is to explicitly modify xcom relating to `return_values`.
        We do this so that the `XComArg` object can work with `expand` function.

        2. So that we can introduce a keyword argument named `mapper_arr`.
        This keyword argument can help as a dummy argument for the `KubernetesPodOperator.partial().expand` method. Any Airflow Operator can be dynamically mapped to runtime artifacts using `Operator.partial(**kwargs).extend(**mapper_kwargs)` post the introduction of [Dynamic Task Mapping](https://airflow.apache.org/docs/apache-airflow/stable/concepts/dynamic-task-mapping.html).
        The `expand` function takes keyword arguments taken by the operator.

        ## Why override the `execute` method  ?

        When we dynamically map vanilla Airflow operators with artifacts generated at runtime, we need to pass that information via `XComArg` to a operator's keyword argument in the `expand` [function](https://airflow.apache.org/docs/apache-airflow/stable/concepts/dynamic-task-mapping.html#mapping-over-result-of-classic-operators).
        The `XComArg` object retrieves XCom values for a particular task based on a `key`, the default key being `return_values`.
        Oddly dynamic task mapping [doesn't support XCom values from any other key except](https://github.com/apache/airflow/blob/8a34d25049a060a035d4db4a49cd4a0d0b07fb0b/airflow/models/mappedoperator.py#L150) `return_values`
        The values of XCom passed by the `KubernetesPodOperator` are mapped to the `return_values` XCom key.

        The biggest problem this creates is that the values of the Foreach cardinality are stored inside the dictionary of `return_values` and cannot be accessed trivially like : `XComArg(task)['foreach_key']` since they are resolved during runtime.
        This puts us in a bind since the only xcom we can retrieve is the full dictionary and we cannot pass that as the iterable for the mapper tasks.
        Hence, we inherit the `execute` method and push custom xcom keys (needed by downstream tasks such as metaflow taskids) and modify `return_values` captured from the container whenever a foreach related xcom is passed.
        When we encounter a foreach xcom we resolve the cardinality which is passed to an actual list and return that as `return_values`.
        This is later useful in the `Workflow.compile` where the operator's `expand` method is called and we are able to retrieve the xcom value.
        """
        template_fields = KubernetesPodOperator.template_fields + ('metaflow_pathspec', 'metaflow_run_id', 'metaflow_task_id', 'metaflow_attempt', 'metaflow_step_name', 'metaflow_flow_name')

        def __init__(self, *args, mapper_arr=None, flow_name=None, flow_contains_foreach=False, **kwargs) -> None:
            if False:
                return 10
            super().__init__(*args, **kwargs)
            self.mapper_arr = mapper_arr
            self._flow_name = flow_name
            self._flow_contains_foreach = flow_contains_foreach
            self.metaflow_pathspec = AIRFLOW_MACROS.pathspec(self._flow_name, is_foreach=self._flow_contains_foreach)
            self.metaflow_run_id = AIRFLOW_MACROS.RUN_ID
            self.metaflow_task_id = AIRFLOW_MACROS.create_task_id(self._flow_contains_foreach)
            self.metaflow_attempt = AIRFLOW_MACROS.ATTEMPT
            self.metaflow_step_name = AIRFLOW_MACROS.STEPNAME
            self.metaflow_flow_name = self._flow_name

        def execute(self, context):
            if False:
                for i in range(10):
                    print('nop')
            result = super().execute(context)
            if result is None:
                return
            ti = context['ti']
            if TASK_ID_XCOM_KEY in result:
                ti.xcom_push(key=TASK_ID_XCOM_KEY, value=result[TASK_ID_XCOM_KEY])
            if FOREACH_CARDINALITY_XCOM_KEY in result:
                return list(range(result[FOREACH_CARDINALITY_XCOM_KEY]))
    return MetaflowKubernetesOperator

class AirflowTask(object):

    def __init__(self, name, operator_type='kubernetes', flow_name=None, is_mapper_node=False, flow_contains_foreach=False):
        if False:
            for i in range(10):
                print('nop')
        self.name = name
        self._is_mapper_node = is_mapper_node
        self._operator_args = None
        self._operator_type = operator_type
        self._flow_name = flow_name
        self._flow_contains_foreach = flow_contains_foreach

    @property
    def is_mapper_node(self):
        if False:
            for i in range(10):
                print('nop')
        return self._is_mapper_node

    def set_operator_args(self, **kwargs):
        if False:
            while True:
                i = 10
        self._operator_args = kwargs
        return self

    def _make_sensor(self):
        if False:
            i = 10
            return i + 15
        TaskSensor = _get_sensor(self._operator_type)
        return TaskSensor(task_id=self.name, **_parse_sensor_args(self._operator_type, self._operator_args))

    def to_dict(self):
        if False:
            print('Hello World!')
        return {'name': self.name, 'is_mapper_node': self._is_mapper_node, 'operator_type': self._operator_type, 'operator_args': self._operator_args}

    @classmethod
    def from_dict(cls, task_dict, flow_name=None, flow_contains_foreach=False):
        if False:
            for i in range(10):
                print('nop')
        op_args = {} if 'operator_args' not in task_dict else task_dict['operator_args']
        is_mapper_node = False if 'is_mapper_node' not in task_dict else task_dict['is_mapper_node']
        return cls(task_dict['name'], is_mapper_node=is_mapper_node, operator_type=task_dict['operator_type'] if 'operator_type' in task_dict else 'kubernetes', flow_name=flow_name, flow_contains_foreach=flow_contains_foreach).set_operator_args(**op_args)

    def _kubernetes_task(self):
        if False:
            return 10
        MetaflowKubernetesOperator = get_metaflow_kubernetes_operator()
        k8s_args = _kubernetes_pod_operator_args(self._operator_args)
        return MetaflowKubernetesOperator(flow_name=self._flow_name, flow_contains_foreach=self._flow_contains_foreach, **k8s_args)

    def _kubernetes_mapper_task(self):
        if False:
            while True:
                i = 10
        MetaflowKubernetesOperator = get_metaflow_kubernetes_operator()
        k8s_args = _kubernetes_pod_operator_args(self._operator_args)
        return MetaflowKubernetesOperator.partial(flow_name=self._flow_name, flow_contains_foreach=self._flow_contains_foreach, **k8s_args)

    def to_task(self):
        if False:
            return 10
        if self._operator_type == 'kubernetes':
            if not self.is_mapper_node:
                return self._kubernetes_task()
            else:
                return self._kubernetes_mapper_task()
        elif self._operator_type in SensorNames.get_supported_sensors():
            return self._make_sensor()

class Workflow(object):

    def __init__(self, file_path=None, graph_structure=None, metadata=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self._dag_instantiation_params = AirflowDAGArgs(**kwargs)
        self._file_path = file_path
        self._metadata = metadata
        tree = lambda : defaultdict(tree)
        self.states = tree()
        self.metaflow_params = None
        self.graph_structure = graph_structure

    def set_parameters(self, params):
        if False:
            i = 10
            return i + 15
        self.metaflow_params = params

    def add_state(self, state):
        if False:
            for i in range(10):
                print('nop')
        self.states[state.name] = state

    def to_dict(self):
        if False:
            while True:
                i = 10
        return dict(metadata=self._metadata, graph_structure=self.graph_structure, states={s: v.to_dict() for (s, v) in self.states.items()}, dag_instantiation_params=self._dag_instantiation_params.serialize(), file_path=self._file_path, metaflow_params=self.metaflow_params)

    def to_json(self):
        if False:
            while True:
                i = 10
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data_dict):
        if False:
            for i in range(10):
                print('nop')
        re_cls = cls(file_path=data_dict['file_path'], graph_structure=data_dict['graph_structure'], metadata=data_dict['metadata'])
        re_cls._dag_instantiation_params = AirflowDAGArgs.deserialize(data_dict['dag_instantiation_params'])
        for sd in data_dict['states'].values():
            re_cls.add_state(AirflowTask.from_dict(sd, flow_name=data_dict['metadata']['flow_name']))
        re_cls.set_parameters(data_dict['metaflow_params'])
        return re_cls

    @classmethod
    def from_json(cls, json_string):
        if False:
            for i in range(10):
                print('nop')
        data = json.loads(json_string)
        return cls.from_dict(data)

    def _construct_params(self):
        if False:
            for i in range(10):
                print('nop')
        from airflow.models.param import Param
        if self.metaflow_params is None:
            return {}
        param_dict = {}
        for p in self.metaflow_params:
            name = p['name']
            del p['name']
            param_dict[name] = Param(**p)
        return param_dict

    def compile(self):
        if False:
            for i in range(10):
                print('nop')
        from airflow import DAG
        XComArg = get_xcom_arg_class()
        _validate_minimum_airflow_version()
        if self._metadata['contains_foreach']:
            _validate_dynamic_mapping_compatibility()
            _check_foreach_compatible_kubernetes_provider()
        params_dict = self._construct_params()
        dag = DAG(params=params_dict, **self._dag_instantiation_params.arguments)
        dag.fileloc = self._file_path if self._file_path is not None else dag.fileloc

        def add_node(node, parents, dag):
            if False:
                while True:
                    i = 10
            '\n            A recursive function to traverse the specialized\n            graph_structure datastructure.\n            '
            if type(node) == str:
                task = self.states[node].to_task()
                if parents:
                    for parent in parents:
                        if self.states[node].is_mapper_node:
                            task = task.expand(mapper_arr=XComArg(parent))
                        parent >> task
                return [task]
            if type(node) == list:
                if all((isinstance(n, list) for n in node)):
                    curr_parents = parents
                    parent_list = []
                    for node_list in node:
                        last_parent = add_node(node_list, curr_parents, dag)
                        parent_list.extend(last_parent)
                    return parent_list
                else:
                    curr_parents = parents
                    for node_x in node:
                        curr_parents = add_node(node_x, curr_parents, dag)
                    return curr_parents
        with dag:
            parent = None
            for node in self.graph_structure:
                parent = add_node(node, parent, dag)
        return dag