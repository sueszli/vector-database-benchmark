from typing import Dict, List, cast
from dagster import AssetCheckSpec, Definitions, asset, asset_check, graph, job, op, repository, resource, schedule, sensor
from dagster._config.field_utils import EnvVar
from dagster._config.pythonic_config import Config, ConfigurableResource
from dagster._core.definitions.assets_job import build_assets_job
from dagster._core.definitions.events import AssetKey
from dagster._core.definitions.repository_definition import PendingRepositoryDefinition, RepositoryDefinition
from dagster._core.definitions.resource_annotation import ResourceParam
from dagster._core.definitions.resource_definition import ResourceDefinition
from dagster._core.execution.context.init import InitResourceContext
from dagster._core.host_representation import ExternalJobData, external_repository_data_from_def
from dagster._core.host_representation.external_data import ExternalResourceData, NestedResource, NestedResourceType, ResourceJobUsageEntry
from dagster._core.snap import JobSnapshot

def test_repository_snap_all_props():
    if False:
        return 10

    @op
    def noop_op(_):
        if False:
            print('Hello World!')
        pass

    @job
    def noop_job():
        if False:
            i = 10
            return i + 15
        noop_op()

    @repository
    def noop_repo():
        if False:
            while True:
                i = 10
        return [noop_job]
    external_repo_data = external_repository_data_from_def(noop_repo)
    assert external_repo_data.name == 'noop_repo'
    assert len(external_repo_data.external_job_datas) == 1
    assert isinstance(external_repo_data.external_job_datas[0], ExternalJobData)
    job_snapshot = external_repo_data.external_job_datas[0].job_snapshot
    assert isinstance(job_snapshot, JobSnapshot)
    assert job_snapshot.name == 'noop_job'
    assert job_snapshot.description is None
    assert job_snapshot.tags == {}

def resolve_pending_repo_if_required(definitions: Definitions) -> RepositoryDefinition:
    if False:
        return 10
    repo_or_caching_repo = definitions.get_inner_repository_for_loading_process()
    return repo_or_caching_repo.compute_repository_definition() if isinstance(repo_or_caching_repo, PendingRepositoryDefinition) else repo_or_caching_repo

def test_repository_snap_definitions_resources_basic():
    if False:
        print('Hello World!')

    @asset
    def my_asset(foo: ResourceParam[str]):
        if False:
            for i in range(10):
                print('nop')
        pass
    defs = Definitions(assets=[my_asset], resources={'foo': ResourceDefinition.hardcoded_resource('wrapped')})
    repo = resolve_pending_repo_if_required(defs)
    external_repo_data = external_repository_data_from_def(repo)
    assert len(external_repo_data.external_resource_data) == 1
    assert external_repo_data.external_resource_data[0].name == 'foo'
    assert external_repo_data.external_resource_data[0].resource_snapshot.name == 'foo'
    assert external_repo_data.external_resource_data[0].resource_snapshot.description is None
    assert external_repo_data.external_resource_data[0].configured_values == {}

def test_repository_snap_definitions_resources_nested() -> None:
    if False:
        for i in range(10):
            print('nop')

    class MyInnerResource(ConfigurableResource):
        a_str: str

    class MyOuterResource(ConfigurableResource):
        inner: MyInnerResource
    inner = MyInnerResource(a_str='wrapped')
    defs = Definitions(resources={'foo': MyOuterResource(inner=inner)})
    repo = resolve_pending_repo_if_required(defs)
    external_repo_data = external_repository_data_from_def(repo)
    assert external_repo_data.external_resource_data
    assert len(external_repo_data.external_resource_data) == 1
    foo = [data for data in external_repo_data.external_resource_data if data.name == 'foo']
    assert len(foo) == 1
    assert foo[0].resource_type == 'dagster_tests.core_tests.snap_tests.test_repository_snap.test_repository_snap_definitions_resources_nested.<locals>.MyOuterResource'
    assert len(foo[0].nested_resources) == 1
    assert 'inner' in foo[0].nested_resources
    assert foo[0].nested_resources['inner'] == NestedResource(NestedResourceType.ANONYMOUS, 'MyInnerResource')

def test_repository_snap_definitions_resources_nested_top_level() -> None:
    if False:
        i = 10
        return i + 15

    class MyInnerResource(ConfigurableResource):
        a_str: str

    class MyOuterResource(ConfigurableResource):
        inner: MyInnerResource
    inner = MyInnerResource(a_str='wrapped')
    defs = Definitions(resources={'foo': MyOuterResource(inner=inner), 'inner': inner})
    repo = resolve_pending_repo_if_required(defs)
    external_repo_data = external_repository_data_from_def(repo)
    assert external_repo_data.external_resource_data
    assert len(external_repo_data.external_resource_data) == 2
    foo = [data for data in external_repo_data.external_resource_data if data.name == 'foo']
    inner = [data for data in external_repo_data.external_resource_data if data.name == 'inner']
    assert len(foo) == 1
    assert len(inner) == 1
    assert len(foo[0].nested_resources) == 1
    assert 'inner' in foo[0].nested_resources
    assert foo[0].nested_resources['inner'] == NestedResource(NestedResourceType.TOP_LEVEL, 'inner')
    assert foo[0].resource_type == 'dagster_tests.core_tests.snap_tests.test_repository_snap.test_repository_snap_definitions_resources_nested_top_level.<locals>.MyOuterResource'
    assert len(inner[0].parent_resources) == 1
    assert 'foo' in inner[0].parent_resources
    assert inner[0].parent_resources['foo'] == 'inner'
    assert inner[0].resource_type == 'dagster_tests.core_tests.snap_tests.test_repository_snap.test_repository_snap_definitions_resources_nested_top_level.<locals>.MyInnerResource'

def test_repository_snap_definitions_function_style_resources_nested() -> None:
    if False:
        return 10

    @resource
    def my_inner_resource() -> str:
        if False:
            for i in range(10):
                print('nop')
        return 'foo'

    @resource(required_resource_keys={'inner'})
    def my_outer_resource(context: InitResourceContext) -> str:
        if False:
            i = 10
            return i + 15
        return context.resources.inner + 'bar'
    defs = Definitions(resources={'foo': my_outer_resource, 'inner': my_inner_resource})
    repo = resolve_pending_repo_if_required(defs)
    external_repo_data = external_repository_data_from_def(repo)
    assert external_repo_data.external_resource_data
    assert len(external_repo_data.external_resource_data) == 2
    foo = [data for data in external_repo_data.external_resource_data if data.name == 'foo']
    inner = [data for data in external_repo_data.external_resource_data if data.name == 'inner']
    assert len(foo) == 1
    assert len(inner) == 1
    assert len(foo[0].nested_resources) == 1
    assert 'inner' in foo[0].nested_resources
    assert foo[0].nested_resources['inner'] == NestedResource(NestedResourceType.TOP_LEVEL, 'inner')
    assert foo[0].resource_type == 'dagster_tests.core_tests.snap_tests.test_repository_snap.my_outer_resource'
    assert len(inner[0].parent_resources) == 1
    assert 'foo' in inner[0].parent_resources
    assert inner[0].parent_resources['foo'] == 'inner'
    assert inner[0].resource_type == 'dagster_tests.core_tests.snap_tests.test_repository_snap.my_inner_resource'

def test_repository_snap_definitions_resources_nested_many() -> None:
    if False:
        print('Hello World!')

    class MyInnerResource(ConfigurableResource):
        a_str: str

    class MyOuterResource(ConfigurableResource):
        inner: MyInnerResource

    class MyOutermostResource(ConfigurableResource):
        inner: MyOuterResource
    inner = MyInnerResource(a_str='wrapped')
    outer = MyOuterResource(inner=inner)
    defs = Definitions(resources={'outermost': MyOutermostResource(inner=outer), 'outer': outer})
    repo = resolve_pending_repo_if_required(defs)
    external_repo_data = external_repository_data_from_def(repo)
    assert external_repo_data.external_resource_data
    assert len(external_repo_data.external_resource_data) == 2
    outermost = [data for data in external_repo_data.external_resource_data if data.name == 'outermost']
    assert len(outermost) == 1
    assert len(outermost[0].nested_resources) == 1
    assert 'inner' in outermost[0].nested_resources
    assert outermost[0].nested_resources['inner'] == NestedResource(NestedResourceType.TOP_LEVEL, 'outer')
    outer = [data for data in external_repo_data.external_resource_data if data.name == 'outer']
    assert len(outer) == 1
    assert len(outer[0].nested_resources) == 1
    assert 'inner' in outer[0].nested_resources
    assert outer[0].nested_resources['inner'] == NestedResource(NestedResourceType.ANONYMOUS, 'MyInnerResource')

def test_repository_snap_definitions_resources_complex():
    if False:
        for i in range(10):
            print('nop')

    class MyStringResource(ConfigurableResource):
        """My description."""
        my_string: str = 'bar'

    @asset
    def my_asset(foo: MyStringResource):
        if False:
            print('Hello World!')
        pass
    defs = Definitions(assets=[my_asset], resources={'foo': MyStringResource(my_string='baz')})
    repo = resolve_pending_repo_if_required(defs)
    external_repo_data = external_repository_data_from_def(repo)
    assert len(external_repo_data.external_resource_data) == 1
    assert external_repo_data.external_resource_data[0].name == 'foo'
    assert external_repo_data.external_resource_data[0].resource_snapshot.name == 'foo'
    assert external_repo_data.external_resource_data[0].resource_snapshot.description == 'My description.'
    assert len(external_repo_data.external_resource_data[0].config_field_snaps) == 1
    snap = external_repo_data.external_resource_data[0].config_field_snaps[0]
    assert snap.name == 'my_string'
    assert not snap.is_required
    assert snap.default_value_as_json_str == '"bar"'
    assert external_repo_data.external_resource_data[0].configured_values == {'my_string': '"baz"'}

def test_repository_snap_empty():
    if False:
        print('Hello World!')

    @repository
    def empty_repo():
        if False:
            print('Hello World!')
        return []
    external_repo_data = external_repository_data_from_def(empty_repo)
    assert external_repo_data.name == 'empty_repo'
    assert len(external_repo_data.external_job_datas) == 0
    assert len(external_repo_data.external_resource_data) == 0

def test_repository_snap_definitions_env_vars() -> None:
    if False:
        for i in range(10):
            print('nop')

    class MyStringResource(ConfigurableResource):
        my_string: str

    class MyInnerResource(ConfigurableResource):
        my_string: str

    class MyOuterResource(ConfigurableResource):
        inner: MyInnerResource

    class MyInnerConfig(Config):
        my_string: str

    class MyDataStructureResource(ConfigurableResource):
        str_list: List[str]
        str_dict: Dict[str, str]

    class MyResourceWithConfig(ConfigurableResource):
        config: MyInnerConfig
        config_list: List[MyInnerConfig]

    @asset
    def my_asset(foo: MyStringResource):
        if False:
            print('Hello World!')
        pass
    defs = Definitions(assets=[my_asset], resources={'foo': MyStringResource(my_string=EnvVar('MY_STRING')), 'bar': MyStringResource(my_string=EnvVar('MY_STRING')), 'baz': MyStringResource(my_string=EnvVar('MY_OTHER_STRING')), 'qux': MyOuterResource(inner=MyInnerResource(my_string=EnvVar('MY_INNER_STRING'))), 'quux': MyDataStructureResource(str_list=[EnvVar('MY_STRING')], str_dict={'foo': EnvVar('MY_STRING'), 'bar': EnvVar('MY_OTHER_STRING')}), 'quuz': MyResourceWithConfig(config=MyInnerConfig(my_string=EnvVar('MY_CONFIG_NESTED_STRING')), config_list=[MyInnerConfig(my_string=EnvVar('MY_CONFIG_LIST_NESTED_STRING'))])})
    repo = resolve_pending_repo_if_required(defs)
    external_repo_data = external_repository_data_from_def(repo)
    assert external_repo_data.utilized_env_vars
    env_vars = dict(external_repo_data.utilized_env_vars)
    assert len(env_vars) == 5
    assert 'MY_STRING' in env_vars
    assert {consumer.name for consumer in env_vars['MY_STRING']} == {'foo', 'bar', 'quux'}
    assert 'MY_OTHER_STRING' in env_vars
    assert {consumer.name for consumer in env_vars['MY_OTHER_STRING']} == {'baz', 'quux'}
    assert 'MY_INNER_STRING' in env_vars
    assert {consumer.name for consumer in env_vars['MY_INNER_STRING']} == {'qux'}
    assert 'MY_CONFIG_NESTED_STRING' in env_vars
    assert {consumer.name for consumer in env_vars['MY_CONFIG_NESTED_STRING']} == {'quuz'}
    assert 'MY_CONFIG_LIST_NESTED_STRING' in env_vars
    assert {consumer.name for consumer in env_vars['MY_CONFIG_LIST_NESTED_STRING']} == {'quuz'}

def test_repository_snap_definitions_resources_assets_usage() -> None:
    if False:
        return 10

    class MyResource(ConfigurableResource):
        a_str: str

    @asset
    def my_asset(foo: MyResource):
        if False:
            return 10
        pass

    @asset
    def my_other_asset(foo: MyResource, bar: MyResource):
        if False:
            while True:
                i = 10
        pass

    @asset
    def my_third_asset():
        if False:
            i = 10
            return i + 15
        pass
    defs = Definitions(assets=[my_asset, my_other_asset, my_third_asset], resources={'foo': MyResource(a_str='foo'), 'bar': MyResource(a_str='bar'), 'baz': MyResource(a_str='baz')})
    repo = resolve_pending_repo_if_required(defs)
    external_repo_data = external_repository_data_from_def(repo)
    assert external_repo_data.external_resource_data
    assert len(external_repo_data.external_resource_data) == 3
    foo = [data for data in external_repo_data.external_resource_data if data.name == 'foo']
    assert len(foo) == 1
    assert sorted(foo[0].asset_keys_using, key=lambda k: ''.join(k.path)) == [AssetKey('my_asset'), AssetKey('my_other_asset')]
    bar = [data for data in external_repo_data.external_resource_data if data.name == 'bar']
    assert len(bar) == 1
    assert bar[0].asset_keys_using == [AssetKey('my_other_asset')]
    baz = [data for data in external_repo_data.external_resource_data if data.name == 'baz']
    assert len(baz) == 1
    assert baz[0].asset_keys_using == []

def test_repository_snap_definitions_function_style_resources_assets_usage() -> None:
    if False:
        i = 10
        return i + 15

    @resource
    def my_resource() -> str:
        if False:
            i = 10
            return i + 15
        return 'foo'

    @asset
    def my_asset(foo: ResourceParam[str]):
        if False:
            while True:
                i = 10
        pass

    @asset
    def my_other_asset(foo: ResourceParam[str]):
        if False:
            for i in range(10):
                print('nop')
        pass

    @asset
    def my_third_asset():
        if False:
            return 10
        pass
    defs = Definitions(assets=[my_asset, my_other_asset, my_third_asset], resources={'foo': my_resource})
    repo = resolve_pending_repo_if_required(defs)
    external_repo_data = external_repository_data_from_def(repo)
    assert external_repo_data.external_resource_data
    assert len(external_repo_data.external_resource_data) == 1
    foo = external_repo_data.external_resource_data[0]
    assert sorted(foo.asset_keys_using, key=lambda k: ''.join(k.path)) == [AssetKey('my_asset'), AssetKey('my_other_asset')]

def _to_dict(entries: List[ResourceJobUsageEntry]) -> Dict[str, List[str]]:
    if False:
        print('Hello World!')
    return {entry.job_name: sorted([handle.to_string() for handle in entry.node_handles]) for entry in entries}

def test_repository_snap_definitions_resources_job_op_usage() -> None:
    if False:
        for i in range(10):
            print('nop')

    class MyResource(ConfigurableResource):
        a_str: str

    @op
    def my_op(foo: MyResource):
        if False:
            return 10
        pass

    @op
    def my_other_op(foo: MyResource, bar: MyResource):
        if False:
            i = 10
            return i + 15
        pass

    @op
    def my_third_op():
        if False:
            while True:
                i = 10
        pass

    @op
    def my_op_in_other_job(foo: MyResource):
        if False:
            print('Hello World!')
        pass

    @job
    def my_first_job() -> None:
        if False:
            for i in range(10):
                print('nop')
        my_op()
        my_other_op()
        my_third_op()

    @job
    def my_second_job() -> None:
        if False:
            return 10
        my_op_in_other_job()
        my_op_in_other_job()
    defs = Definitions(jobs=[my_first_job, my_second_job], resources={'foo': MyResource(a_str='foo'), 'bar': MyResource(a_str='bar')})
    repo = resolve_pending_repo_if_required(defs)
    external_repo_data = external_repository_data_from_def(repo)
    assert external_repo_data.external_resource_data
    assert len(external_repo_data.external_resource_data) == 2
    foo = [data for data in external_repo_data.external_resource_data if data.name == 'foo']
    assert len(foo) == 1
    assert _to_dict(foo[0].job_ops_using) == {'my_first_job': ['my_op', 'my_other_op'], 'my_second_job': ['my_op_in_other_job', 'my_op_in_other_job_2']}
    bar = [data for data in external_repo_data.external_resource_data if data.name == 'bar']
    assert len(bar) == 1
    assert _to_dict(bar[0].job_ops_using) == {'my_first_job': ['my_other_op']}

def test_repository_snap_definitions_resources_job_op_usage_graph() -> None:
    if False:
        i = 10
        return i + 15

    class MyResource(ConfigurableResource):
        a_str: str

    @op
    def my_op(foo: MyResource):
        if False:
            print('Hello World!')
        pass

    @op
    def my_other_op(foo: MyResource, bar: MyResource):
        if False:
            for i in range(10):
                print('nop')
        pass

    @graph
    def my_graph():
        if False:
            i = 10
            return i + 15
        my_op()
        my_other_op()

    @op
    def my_third_op(foo: MyResource):
        if False:
            for i in range(10):
                print('nop')
        pass

    @graph
    def my_other_graph():
        if False:
            print('Hello World!')
        my_third_op()

    @job
    def my_job() -> None:
        if False:
            i = 10
            return i + 15
        my_graph()
        my_other_graph()
        my_op()
        my_op()
    defs = Definitions(jobs=[my_job], resources={'foo': MyResource(a_str='foo'), 'bar': MyResource(a_str='bar')})
    repo = resolve_pending_repo_if_required(defs)
    external_repo_data = external_repository_data_from_def(repo)
    assert external_repo_data.external_resource_data
    assert len(external_repo_data.external_resource_data) == 2
    foo = [data for data in external_repo_data.external_resource_data if data.name == 'foo']
    assert len(foo) == 1
    assert _to_dict(foo[0].job_ops_using) == {'my_job': ['my_graph.my_op', 'my_graph.my_other_op', 'my_op', 'my_op_2', 'my_other_graph.my_third_op']}
    bar = [data for data in external_repo_data.external_resource_data if data.name == 'bar']
    assert len(bar) == 1
    assert _to_dict(bar[0].job_ops_using) == {'my_job': ['my_graph.my_other_op']}

def test_asset_check():
    if False:
        for i in range(10):
            print('nop')

    @asset
    def my_asset():
        if False:
            i = 10
            return i + 15
        pass

    @asset_check(asset=my_asset)
    def my_asset_check():
        if False:
            while True:
                i = 10
        ...

    @asset_check(asset=my_asset)
    def my_asset_check_2():
        if False:
            return 10
        ...
    defs = Definitions(assets=[my_asset], asset_checks=[my_asset_check, my_asset_check_2])
    repo = resolve_pending_repo_if_required(defs)
    external_repo_data = external_repository_data_from_def(repo)
    assert len(external_repo_data.external_asset_checks) == 2
    assert external_repo_data.external_asset_checks[0].name == 'my_asset_check'
    assert external_repo_data.external_asset_checks[1].name == 'my_asset_check_2'

def test_asset_check_in_asset_op():
    if False:
        i = 10
        return i + 15

    @asset(check_specs=[AssetCheckSpec(name='my_other_asset_check', asset='my_asset'), AssetCheckSpec(name='my_other_asset_check_2', asset='my_asset')])
    def my_asset():
        if False:
            i = 10
            return i + 15
        pass

    @asset_check(asset=my_asset)
    def my_asset_check():
        if False:
            i = 10
            return i + 15
        ...
    defs = Definitions(assets=[my_asset], asset_checks=[my_asset_check])
    repo = resolve_pending_repo_if_required(defs)
    external_repo_data = external_repository_data_from_def(repo)
    assert len(external_repo_data.external_asset_checks) == 3
    assert external_repo_data.external_asset_checks[0].name == 'my_asset_check'
    assert external_repo_data.external_asset_checks[1].name == 'my_other_asset_check'
    assert external_repo_data.external_asset_checks[2].name == 'my_other_asset_check_2'

def test_asset_check_multiple_jobs():
    if False:
        i = 10
        return i + 15

    @asset(check_specs=[AssetCheckSpec(name='my_other_asset_check', asset='my_asset')])
    def my_asset():
        if False:
            return 10
        pass

    @asset_check(asset=my_asset)
    def my_asset_check():
        if False:
            print('Hello World!')
        ...
    my_job = build_assets_job('my_job', [my_asset])
    defs = Definitions(assets=[my_asset], asset_checks=[my_asset_check], jobs=[my_job])
    repo = resolve_pending_repo_if_required(defs)
    external_repo_data = external_repository_data_from_def(repo)
    assert len(external_repo_data.external_asset_checks) == 2
    assert external_repo_data.external_asset_checks[0].name == 'my_asset_check'
    assert external_repo_data.external_asset_checks[1].name == 'my_other_asset_check'

def test_repository_snap_definitions_resources_schedule_sensor_usage():
    if False:
        return 10

    class MyResource(ConfigurableResource):
        a_str: str

    @op
    def my_op() -> None:
        if False:
            for i in range(10):
                print('nop')
        pass

    @job
    def my_job() -> None:
        if False:
            i = 10
            return i + 15
        my_op()

    @sensor(job=my_job)
    def my_sensor(foo: MyResource):
        if False:
            return 10
        pass

    @sensor(job=my_job)
    def my_sensor_two(foo: MyResource, bar: MyResource):
        if False:
            while True:
                i = 10
        pass

    @schedule(job=my_job, cron_schedule='* * * * *')
    def my_schedule(foo: MyResource):
        if False:
            for i in range(10):
                print('nop')
        pass

    @schedule(job=my_job, cron_schedule='* * * * *')
    def my_schedule_two(foo: MyResource, baz: MyResource):
        if False:
            return 10
        pass
    defs = Definitions(resources={'foo': MyResource(a_str='foo'), 'bar': MyResource(a_str='bar'), 'baz': MyResource(a_str='baz')}, sensors=[my_sensor, my_sensor_two], schedules=[my_schedule, my_schedule_two])
    repo = resolve_pending_repo_if_required(defs)
    external_repo_data = external_repository_data_from_def(repo)
    assert external_repo_data.external_resource_data
    assert len(external_repo_data.external_resource_data) == 3
    foo = [data for data in external_repo_data.external_resource_data if data.name == 'foo']
    assert len(foo) == 1
    assert set(cast(ExternalResourceData, foo[0]).schedules_using) == {'my_schedule', 'my_schedule_two'}
    assert set(cast(ExternalResourceData, foo[0]).sensors_using) == {'my_sensor', 'my_sensor_two'}
    bar = [data for data in external_repo_data.external_resource_data if data.name == 'bar']
    assert len(bar) == 1
    assert set(cast(ExternalResourceData, bar[0]).schedules_using) == set()
    assert set(cast(ExternalResourceData, bar[0]).sensors_using) == {'my_sensor_two'}
    baz = [data for data in external_repo_data.external_resource_data if data.name == 'baz']
    assert len(baz) == 1
    assert set(cast(ExternalResourceData, baz[0]).schedules_using) == set({'my_schedule_two'})
    assert set(cast(ExternalResourceData, baz[0]).sensors_using) == set()