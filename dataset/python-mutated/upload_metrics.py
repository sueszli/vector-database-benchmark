import datetime
import inspect
import os
import time
import uuid
from decimal import Decimal
from typing import Any, Dict
from warnings import warn
EMIT_METRICS = False
try:
    import boto3
    EMIT_METRICS = True
except ImportError as e:
    print(f'Unable to import boto3. Will not be emitting metrics.... Reason: {e}')

class EnvVarMetric:
    name: str
    env_var: str
    required: bool = True
    type_conversion_fn: Any = None

    def __init__(self, name: str, env_var: str, required: bool=True, type_conversion_fn: Any=None) -> None:
        if False:
            print('Hello World!')
        self.name = name
        self.env_var = env_var
        self.required = required
        self.type_conversion_fn = type_conversion_fn

    def value(self) -> Any:
        if False:
            return 10
        value = os.environ.get(self.env_var)
        DEFAULT_ENVVAR_VALUES = [None, '']
        if value in DEFAULT_ENVVAR_VALUES:
            if not self.required:
                return None
            raise ValueError(f'Missing {self.name}. Please set the {self.env_var} environment variable to pass in this value.')
        if self.type_conversion_fn:
            return self.type_conversion_fn(value)
        return value
global_metrics: Dict[str, Any] = {}

def add_global_metric(metric_name: str, metric_value: Any) -> None:
    if False:
        for i in range(10):
            print('nop')
    '\n    Adds stats that should be emitted with every metric by the current process.\n    If the emit_metrics method specifies a metric with the same name, it will\n    overwrite this value.\n    '
    global_metrics[metric_name] = metric_value

def emit_metric(metric_name: str, metrics: Dict[str, Any]) -> None:
    if False:
        return 10
    "\n    Upload a metric to DynamoDB (and from there, Rockset).\n\n    Even if EMIT_METRICS is set to False, this function will still run the code to\n    validate and shape the metrics, skipping just the upload.\n\n    Parameters:\n        metric_name:\n            Name of the metric. Every unique metric should have a different name\n            and be emitted just once per run attempt.\n            Metrics are namespaced by their module and the function that emitted them.\n        metrics: The actual data to record.\n\n    Some default values are populated from environment variables, which must be set\n    for metrics to be emitted. (If they're not set, this function becomes a noop):\n    "
    if metrics is None:
        raise ValueError("You didn't ask to upload any metrics!")
    metrics = {**global_metrics, **metrics}
    env_var_metrics = [EnvVarMetric('repo', 'GITHUB_REPOSITORY'), EnvVarMetric('workflow', 'GITHUB_WORKFLOW'), EnvVarMetric('build_environment', 'BUILD_ENVIRONMENT'), EnvVarMetric('job', 'GITHUB_JOB'), EnvVarMetric('test_config', 'TEST_CONFIG', required=False), EnvVarMetric('pr_number', 'PR_NUMBER', required=False, type_conversion_fn=int), EnvVarMetric('run_id', 'GITHUB_RUN_ID', type_conversion_fn=int), EnvVarMetric('run_number', 'GITHUB_RUN_NUMBER', type_conversion_fn=int), EnvVarMetric('run_attempt', 'GITHUB_RUN_ATTEMPT', type_conversion_fn=int), EnvVarMetric('job_id', 'JOB_ID', type_conversion_fn=int)]
    calling_frame = inspect.currentframe().f_back
    calling_frame_info = inspect.getframeinfo(calling_frame)
    calling_file = os.path.basename(calling_frame_info.filename)
    calling_module = inspect.getmodule(calling_frame).__name__
    calling_function = calling_frame_info.function
    try:
        reserved_metrics = {'metric_name': metric_name, 'calling_file': calling_file, 'calling_module': calling_module, 'calling_function': calling_function, 'timestamp': datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f'), **{m.name: m.value() for m in env_var_metrics if m.value()}}
    except ValueError as e:
        warn(f'Not emitting metrics for {metric_name}. {e}')
        return
    reserved_metrics['dynamo_key'] = f'{metric_name}_{int(time.time())}_{uuid.uuid1().hex}'
    for key in reserved_metrics.keys():
        used_reserved_keys = [k for k in metrics.keys() if k == key]
        if used_reserved_keys:
            raise ValueError(f"Metrics dict contains reserved keys: [{', '.join(key)}]")
    metrics = _convert_float_values_to_decimals(metrics)
    if EMIT_METRICS:
        try:
            session = boto3.Session(region_name='us-east-1')
            session.resource('dynamodb').Table('torchci-metrics').put_item(Item={**reserved_metrics, **metrics})
        except Exception as e:
            warn(f'Error uploading metric {metric_name} to DynamoDB: {e}')
            return
    else:
        print(f"Not emitting metrics for {metric_name}. Boto wasn't imported.")

def _convert_float_values_to_decimals(data: Dict[str, Any]) -> Dict[str, Any]:
    if False:
        for i in range(10):
            print('nop')

    def _helper(o: Any) -> Any:
        if False:
            for i in range(10):
                print('nop')
        if isinstance(o, float):
            return Decimal(str(o))
        if isinstance(o, list):
            return [_helper(v) for v in o]
        if isinstance(o, dict):
            return {_helper(k): _helper(v) for (k, v) in o.items()}
        return o
    return {k: _helper(v) for (k, v) in data.items()}