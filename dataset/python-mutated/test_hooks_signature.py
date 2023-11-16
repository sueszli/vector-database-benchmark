from __future__ import annotations
import inspect
from importlib import import_module
from pathlib import Path
import pytest
from airflow.exceptions import AirflowOptionalProviderFeatureException
from airflow.providers.amazon.aws.hooks.base_aws import AwsGenericHook
BASE_AWS_HOOKS = ['AwsGenericHook', 'AwsBaseHook']
ALLOWED_THICK_HOOKS_PARAMETERS: dict[str, set[str]] = {'AthenaHook': {'sleep_time', 'log_query'}, 'BatchClientHook': {'status_retries', 'max_retries'}, 'BatchWaitersHook': {'waiter_config'}, 'DataSyncHook': {'wait_interval_seconds'}, 'DynamoDBHook': {'table_name', 'table_keys'}, 'EC2Hook': {'api_type'}, 'ElastiCacheReplicationGroupHook': {'exponential_back_off_factor', 'max_retries', 'initial_poke_interval'}, 'EmrHook': {'emr_conn_id'}, 'EmrContainerHook': {'virtual_cluster_id'}, 'FirehoseHook': {'delivery_stream'}, 'GlueJobHook': {'job_name', 'concurrent_run_limit', 'job_poll_interval', 'create_job_kwargs', 'desc', 'iam_role_arn', 's3_bucket', 'iam_role_name', 'update_config', 'retry_limit', 'num_of_dpus', 'script_location'}, 'S3Hook': {'transfer_config_args', 'aws_conn_id', 'extra_args'}}

def get_aws_hooks_modules():
    if False:
        for i in range(10):
            print('nop')
    'Parse Amazon Provider metadata and find all hooks based on `AwsGenericHook` and return it.'
    hooks_dir = Path(__file__).absolute().parents[5] / 'airflow' / 'providers' / 'amazon' / 'aws' / 'hooks'
    if not hooks_dir.exists():
        msg = f'Amazon Provider hooks directory not found: {hooks_dir.__fspath__()!r}'
        raise FileNotFoundError(msg)
    elif not hooks_dir.is_dir():
        raise NotADirectoryError(hooks_dir.__fspath__())
    for module in hooks_dir.glob('*.py'):
        name = module.stem
        if name.startswith('_'):
            continue
        module_string = f'airflow.providers.amazon.aws.hooks.{name}'
        yield pytest.param(module_string, id=name)

def get_aws_hooks_from_module(hook_module: str) -> list[tuple[type[AwsGenericHook], str]]:
    if False:
        print('Hello World!')
    try:
        imported_module = import_module(hook_module)
    except AirflowOptionalProviderFeatureException as ex:
        pytest.skip(str(ex))
    else:
        hooks = []
        for (name, o) in vars(imported_module).items():
            if name in BASE_AWS_HOOKS:
                continue
            if isinstance(o, type) and o.__module__ != 'builtins' and issubclass(o, AwsGenericHook):
                hooks.append((o, name))
        return hooks

def validate_hook(hook: type[AwsGenericHook], hook_name: str, hook_module: str) -> tuple[bool, str | None]:
    if False:
        while True:
            i = 10
    hook_extra_parameters = set()
    for (k, v) in inspect.signature(hook).parameters.items():
        if v.kind == inspect.Parameter.VAR_POSITIONAL:
            k = '*args'
        elif v.kind == inspect.Parameter.VAR_KEYWORD:
            k = '**kwargs'
        hook_extra_parameters.add(k)
    hook_extra_parameters.difference_update({'self', '*args', '**kwargs'})
    allowed_parameters = ALLOWED_THICK_HOOKS_PARAMETERS.get(hook_name, set())
    if allowed_parameters:
        hook_extra_parameters -= allowed_parameters
    if not hook_extra_parameters:
        return (True, None)
    if not allowed_parameters:
        msg = f"'{hook_module}.{hook_name}' has additional attributes {', '.join(map(repr, hook_extra_parameters))}. Expected that all `boto3` related hooks (based on `AwsGenericHook` or `AwsBaseHook`) should not use additional attributes in class constructor, please move them to method signatures. Make sure that {hook_name!r} constructor has signature `def __init__(self, *args, **kwargs):`"
    else:
        msg = f"'{hook_module}.{hook_name}' allowed only {', '.join(map(repr, allowed_parameters))} additional attributes, but got extra parameters {', '.join(map(repr, hook_extra_parameters))}. Please move additional attributes from class constructor into method signatures. "
    return (False, msg)

@pytest.mark.parametrize('hook_module', get_aws_hooks_modules())
def test_expected_thin_hooks(hook_module: str):
    if False:
        while True:
            i = 10
    '\n    Test Amazon provider Hooks\' signatures.\n\n    All hooks should provide thin wrapper around boto3 / aiobotocore,\n    that mean we should not define additional parameters in Hook parameters.\n    It should be defined in appropriate methods.\n\n    .. code-block:: python\n\n        # Bad: Thick wrapper\n        from airflow.providers.amazon.aws.hooks.base_aws import AwsBaseHook\n\n\n        class AwsServiceName(AwsBaseHook):\n            def __init__(self, foo: str, spam: str, *args, **kwargs) -> None:\n                kwargs.update(dict(client_type="service", resource_type=None))\n                super().__init__(*args, **kwargs)\n                self.foo = foo\n                self.spam = spam\n\n            def method1(self):\n                if self.foo == "bar":\n                    ...\n\n            def method2(self):\n                if self.spam == "egg":\n                    ...\n\n    .. code-block:: python\n\n        # Good: Thin wrapper\n        class AwsServiceName(AwsBaseHook):\n            def __init__(self, *args, **kwargs) -> None:\n                kwargs.update(dict(client_type="service", resource_type=None))\n                super().__init__(*args, **kwargs)\n\n            def method1(self, foo: str):\n                if foo == "bar":\n                    ...\n\n            def method2(self, spam: str):\n                if spam == "egg":\n                    ...\n\n    '
    hooks = get_aws_hooks_from_module(hook_module)
    if not hooks:
        pytest.skip(reason=f"Module {hook_module!r} doesn't contain subclasses of `AwsGenericHook`.")
    errors = [message for (valid, message) in (validate_hook(hook, hook_name, hook_module) for (hook, hook_name) in hooks) if not valid and message]
    if errors:
        errors_msg = '\n * '.join(errors)
        pytest.fail(reason=f'Found errors in {hook_module}:\n * {errors_msg}')