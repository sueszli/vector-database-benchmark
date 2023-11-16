from contextlib import contextmanager
from typing import Any, Dict, Generator, Mapping, Optional, cast
import dagster._check as check
from dagster._config import process_config
from dagster._core.definitions.resource_definition import ResourceDefinition, Resources, ScopedResourcesBuilder
from dagster._core.definitions.run_config import define_resource_dictionary_cls
from dagster._core.errors import DagsterInvalidConfigError
from dagster._core.execution.resources_init import resource_initialization_manager
from dagster._core.instance import DagsterInstance
from dagster._core.log_manager import DagsterLogManager
from dagster._core.storage.dagster_run import DagsterRun
from dagster._core.storage.io_manager import IOManager, IOManagerDefinition
from dagster._core.system_config.objects import ResourceConfig, config_map_resources
from .api import ephemeral_instance_if_missing
from .context_creation_job import initialize_console_manager

def get_mapped_resource_config(resource_defs: Mapping[str, ResourceDefinition], resource_config: Mapping[str, Any]) -> Mapping[str, ResourceConfig]:
    if False:
        while True:
            i = 10
    resource_config_schema = define_resource_dictionary_cls(resource_defs, set(resource_defs.keys()))
    config_evr = process_config(resource_config_schema, resource_config)
    if not config_evr.success:
        raise DagsterInvalidConfigError('Error in config for resources ', config_evr.errors, resource_config)
    config_value = cast(Dict[str, Any], config_evr.value)
    return config_map_resources(resource_defs, config_value)

@contextmanager
def build_resources(resources: Mapping[str, Any], instance: Optional[DagsterInstance]=None, resource_config: Optional[Mapping[str, Any]]=None, dagster_run: Optional[DagsterRun]=None, log_manager: Optional[DagsterLogManager]=None) -> Generator[Resources, None, None]:
    if False:
        return 10
    'Context manager that yields resources using provided resource definitions and run config.\n\n    This API allows for using resources in an independent context. Resources will be initialized\n    with the provided run config, and optionally, dagster_run. The resulting resources will be\n    yielded on a dictionary keyed identically to that provided for `resource_defs`. Upon exiting the\n    context, resources will also be torn down safely.\n\n    Args:\n        resources (Mapping[str, Any]): Resource instances or definitions to build. All\n            required resource dependencies to a given resource must be contained within this\n            dictionary, or the resource build will fail.\n        instance (Optional[DagsterInstance]): The dagster instance configured to instantiate\n            resources on.\n        resource_config (Optional[Mapping[str, Any]]): A dict representing the config to be\n            provided to each resource during initialization and teardown.\n        dagster_run (Optional[PipelineRun]): The pipeline run to provide during resource\n            initialization and teardown. If the provided resources require either the `dagster_run`\n            or `run_id` attributes of the provided context during resource initialization and/or\n            teardown, this must be provided, or initialization will fail.\n        log_manager (Optional[DagsterLogManager]): Log Manager to use during resource\n            initialization. Defaults to system log manager.\n\n    Examples:\n        .. code-block:: python\n\n            from dagster import resource, build_resources\n\n            @resource\n            def the_resource():\n                return "foo"\n\n            with build_resources(resources={"from_def": the_resource, "from_val": "bar"}) as resources:\n                assert resources.from_def == "foo"\n                assert resources.from_val == "bar"\n\n    '
    resources = check.mapping_param(resources, 'resource_defs', key_type=str)
    instance = check.opt_inst_param(instance, 'instance', DagsterInstance)
    resource_config = check.opt_mapping_param(resource_config, 'resource_config', key_type=str)
    log_manager = check.opt_inst_param(log_manager, 'log_manager', DagsterLogManager)
    resource_defs = wrap_resources_for_execution(resources)
    mapped_resource_config = get_mapped_resource_config(resource_defs, resource_config)
    with ephemeral_instance_if_missing(instance) as dagster_instance:
        resources_manager = resource_initialization_manager(resource_defs=resource_defs, resource_configs=mapped_resource_config, log_manager=log_manager if log_manager else initialize_console_manager(dagster_run), execution_plan=None, dagster_run=dagster_run, resource_keys_to_init=set(resource_defs.keys()), instance=dagster_instance, emit_persistent_events=False)
        try:
            list(resources_manager.generate_setup_events())
            instantiated_resources = check.inst(resources_manager.get_object(), ScopedResourcesBuilder)
            yield instantiated_resources.build(set(instantiated_resources.resource_instance_dict.keys()))
        finally:
            list(resources_manager.generate_teardown_events())

def wrap_resources_for_execution(resources: Optional[Mapping[str, Any]]=None) -> Dict[str, ResourceDefinition]:
    if False:
        while True:
            i = 10
    return {resource_key: wrap_resource_for_execution(resource) for (resource_key, resource) in resources.items()} if resources else {}

def wrap_resource_for_execution(resource: Any) -> ResourceDefinition:
    if False:
        i = 10
        return i + 15
    from dagster._config.pythonic_config import ConfigurableResourceFactory, PartialResource
    if isinstance(resource, (ConfigurableResourceFactory, PartialResource)):
        return resource.get_resource_definition()
    elif isinstance(resource, ResourceDefinition):
        return resource
    elif isinstance(resource, IOManager):
        return IOManagerDefinition.hardcoded_io_manager(resource)
    else:
        return ResourceDefinition.hardcoded_resource(resource)