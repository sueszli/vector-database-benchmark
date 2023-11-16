import collections.abc
import inspect
from functools import update_wrapper
from typing import Any, Callable, Optional, Sequence, Set, Union
import dagster._check as check
from dagster._annotations import experimental
from dagster._core.definitions.asset_selection import AssetSelection
from ...errors import DagsterInvariantViolationError
from ..asset_sensor_definition import AssetSensorDefinition
from ..events import AssetKey
from ..multi_asset_sensor_definition import AssetMaterializationFunction, MultiAssetMaterializationFunction, MultiAssetSensorDefinition
from ..run_request import SensorResult
from ..sensor_definition import DefaultSensorStatus, RawSensorEvaluationFunction, RunRequest, SensorDefinition, SkipReason
from ..target import ExecutableDefinition

def sensor(job_name: Optional[str]=None, *, name: Optional[str]=None, minimum_interval_seconds: Optional[int]=None, description: Optional[str]=None, job: Optional[ExecutableDefinition]=None, jobs: Optional[Sequence[ExecutableDefinition]]=None, default_status: DefaultSensorStatus=DefaultSensorStatus.STOPPED, asset_selection: Optional[AssetSelection]=None, required_resource_keys: Optional[Set[str]]=None) -> Callable[[RawSensorEvaluationFunction], SensorDefinition]:
    if False:
        for i in range(10):
            print('nop')
    "Creates a sensor where the decorated function is used as the sensor's evaluation function.\n\n    The decorated function may:\n\n    1. Return a `RunRequest` object.\n    2. Return a list of `RunRequest` objects.\n    3. Return a `SkipReason` object, providing a descriptive message of why no runs were requested.\n    4. Return nothing (skipping without providing a reason)\n    5. Yield a `SkipReason` or yield one or more `RunRequest` objects.\n\n    Takes a :py:class:`~dagster.SensorEvaluationContext`.\n\n    Args:\n        name (Optional[str]): The name of the sensor. Defaults to the name of the decorated\n            function.\n        minimum_interval_seconds (Optional[int]): The minimum number of seconds that will elapse\n            between sensor evaluations.\n        description (Optional[str]): A human-readable description of the sensor.\n        job (Optional[Union[GraphDefinition, JobDefinition, UnresolvedAssetJobDefinition]]):\n            The job to be executed when the sensor fires.\n        jobs (Optional[Sequence[Union[GraphDefinition, JobDefinition, UnresolvedAssetJobDefinition]]]):\n            (experimental) A list of jobs to be executed when the sensor fires.\n        default_status (DefaultSensorStatus): Whether the sensor starts as running or not. The default\n            status can be overridden from the Dagster UI or via the GraphQL API.\n        asset_selection (AssetSelection): (Experimental) an asset selection to launch a run for if\n            the sensor condition is met. This can be provided instead of specifying a job.\n    "
    check.opt_str_param(name, 'name')

    def inner(fn: RawSensorEvaluationFunction) -> SensorDefinition:
        if False:
            print('Hello World!')
        check.callable_param(fn, 'fn')
        sensor_def = SensorDefinition.dagster_internal_init(name=name, job_name=job_name, evaluation_fn=fn, minimum_interval_seconds=minimum_interval_seconds, description=description, job=job, jobs=jobs, default_status=default_status, asset_selection=asset_selection, required_resource_keys=required_resource_keys)
        update_wrapper(sensor_def, wrapped=fn)
        return sensor_def
    return inner

def asset_sensor(asset_key: AssetKey, *, job_name: Optional[str]=None, name: Optional[str]=None, minimum_interval_seconds: Optional[int]=None, description: Optional[str]=None, job: Optional[ExecutableDefinition]=None, jobs: Optional[Sequence[ExecutableDefinition]]=None, default_status: DefaultSensorStatus=DefaultSensorStatus.STOPPED, required_resource_keys: Optional[Set[str]]=None) -> Callable[[AssetMaterializationFunction], AssetSensorDefinition]:
    if False:
        i = 10
        return i + 15
    'Creates an asset sensor where the decorated function is used as the asset sensor\'s evaluation\n    function.\n\n    If the asset has been materialized multiple times between since the last sensor tick, the\n    evaluation function will only be invoked once, with the latest materialization.\n\n    The decorated function may:\n\n    1. Return a `RunRequest` object.\n    2. Return a list of `RunRequest` objects.\n    3. Return a `SkipReason` object, providing a descriptive message of why no runs were requested.\n    4. Return nothing (skipping without providing a reason)\n    5. Yield a `SkipReason` or yield one or more `RunRequest` objects.\n\n    Takes a :py:class:`~dagster.SensorEvaluationContext` and an EventLogEntry corresponding to an\n    AssetMaterialization event.\n\n    Args:\n        asset_key (AssetKey): The asset_key this sensor monitors.\n        name (Optional[str]): The name of the sensor. Defaults to the name of the decorated\n            function.\n        minimum_interval_seconds (Optional[int]): The minimum number of seconds that will elapse\n            between sensor evaluations.\n        description (Optional[str]): A human-readable description of the sensor.\n        job (Optional[Union[GraphDefinition, JobDefinition, UnresolvedAssetJobDefinition]]): The\n            job to be executed when the sensor fires.\n        jobs (Optional[Sequence[Union[GraphDefinition, JobDefinition, UnresolvedAssetJobDefinition]]]):\n            (experimental) A list of jobs to be executed when the sensor fires.\n        default_status (DefaultSensorStatus): Whether the sensor starts as running or not. The default\n            status can be overridden from the Dagster UI or via the GraphQL API.\n\n\n    Example:\n        .. code-block:: python\n\n            from dagster import AssetKey, EventLogEntry, SensorEvaluationContext, asset_sensor\n\n\n            @asset_sensor(asset_key=AssetKey("my_table"), job=my_job)\n            def my_asset_sensor(context: SensorEvaluationContext, asset_event: EventLogEntry):\n                return RunRequest(\n                    run_key=context.cursor,\n                    run_config={\n                        "ops": {\n                            "read_materialization": {\n                                "config": {\n                                    "asset_key": asset_event.dagster_event.asset_key.path,\n                                }\n                            }\n                        }\n                    },\n                )\n    '
    check.opt_str_param(name, 'name')

    def inner(fn: AssetMaterializationFunction) -> AssetSensorDefinition:
        if False:
            for i in range(10):
                print('nop')
        check.callable_param(fn, 'fn')
        sensor_name = name or fn.__name__

        def _wrapped_fn(*args, **kwargs) -> Any:
            if False:
                return 10
            result = fn(*args, **kwargs)
            if inspect.isgenerator(result) or isinstance(result, list):
                for item in result:
                    yield item
            elif isinstance(result, (RunRequest, SkipReason)):
                yield result
            elif isinstance(result, SensorResult):
                if result.cursor:
                    raise DagsterInvariantViolationError(f'Error in asset sensor {sensor_name}: Sensor returned a SensorResult with a cursor value. The cursor is managed by the asset sensor and should not be modified by a user.')
                yield result
            elif result is not None:
                raise DagsterInvariantViolationError(f'Error in sensor {sensor_name}: Sensor unexpectedly returned output {result} of type {type(result)}.  Should only return SkipReason or RunRequest objects.')
        _wrapped_fn = update_wrapper(_wrapped_fn, wrapped=fn)
        return AssetSensorDefinition(name=sensor_name, asset_key=asset_key, job_name=job_name, asset_materialization_fn=_wrapped_fn, minimum_interval_seconds=minimum_interval_seconds, description=description, job=job, jobs=jobs, default_status=default_status, required_resource_keys=required_resource_keys)
    return inner

@experimental
def multi_asset_sensor(monitored_assets: Union[Sequence[AssetKey], AssetSelection], *, job_name: Optional[str]=None, name: Optional[str]=None, minimum_interval_seconds: Optional[int]=None, description: Optional[str]=None, job: Optional[ExecutableDefinition]=None, jobs: Optional[Sequence[ExecutableDefinition]]=None, default_status: DefaultSensorStatus=DefaultSensorStatus.STOPPED, request_assets: Optional[AssetSelection]=None, required_resource_keys: Optional[Set[str]]=None) -> Callable[[MultiAssetMaterializationFunction], MultiAssetSensorDefinition]:
    if False:
        return 10
    "Creates an asset sensor that can monitor multiple assets.\n\n    The decorated function is used as the asset sensor's evaluation\n    function.  The decorated function may:\n\n    1. Return a `RunRequest` object.\n    2. Return a list of `RunRequest` objects.\n    3. Return a `SkipReason` object, providing a descriptive message of why no runs were requested.\n    4. Return nothing (skipping without providing a reason)\n    5. Yield a `SkipReason` or yield one or more `RunRequest` objects.\n\n    Takes a :py:class:`~dagster.MultiAssetSensorEvaluationContext`.\n\n    Args:\n        monitored_assets (Union[Sequence[AssetKey], AssetSelection]): The assets this\n            sensor monitors. If an AssetSelection object is provided, it will only apply to assets\n            within the Definitions that this sensor is part of.\n        name (Optional[str]): The name of the sensor. Defaults to the name of the decorated\n            function.\n        minimum_interval_seconds (Optional[int]): The minimum number of seconds that will elapse\n            between sensor evaluations.\n        description (Optional[str]): A human-readable description of the sensor.\n        job (Optional[Union[GraphDefinition, JobDefinition, UnresolvedAssetJobDefinition]]): The\n            job to be executed when the sensor fires.\n        jobs (Optional[Sequence[Union[GraphDefinition, JobDefinition, UnresolvedAssetJobDefinition]]]):\n            (experimental) A list of jobs to be executed when the sensor fires.\n        default_status (DefaultSensorStatus): Whether the sensor starts as running or not. The default\n            status can be overridden from the Dagster UI or via the GraphQL API.\n        request_assets (Optional[AssetSelection]): (Experimental) an asset selection to launch a run\n            for if the sensor condition is met. This can be provided instead of specifying a job.\n    "
    check.opt_str_param(name, 'name')
    if not isinstance(monitored_assets, AssetSelection) and (not (isinstance(monitored_assets, collections.abc.Sequence) and all((isinstance(el, AssetKey) for el in monitored_assets)))):
        check.failed(f'The value passed to monitored_assets param must be either an AssetSelection or a Sequence of AssetKeys, but was a {type(monitored_assets)}')

    def inner(fn: MultiAssetMaterializationFunction) -> MultiAssetSensorDefinition:
        if False:
            print('Hello World!')
        check.callable_param(fn, 'fn')
        sensor_name = name or fn.__name__
        sensor_def = MultiAssetSensorDefinition(name=sensor_name, monitored_assets=monitored_assets, job_name=job_name, asset_materialization_fn=fn, minimum_interval_seconds=minimum_interval_seconds, description=description, job=job, jobs=jobs, default_status=default_status, request_assets=request_assets, required_resource_keys=required_resource_keys)
        update_wrapper(sensor_def, wrapped=fn)
        return sensor_def
    return inner