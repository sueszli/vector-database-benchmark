import re
import pytest
from dagster import AssetExecutionContext, AssetKey, AssetsDefinition, DagsterInvalidDefinitionError, Definitions, ResourceDefinition, ScheduleDefinition, SourceAsset, asset, build_schedule_from_partitioned_job, create_repository_using_definitions_args, define_asset_job, graph, in_process_executor, materialize, mem_io_manager, op, repository, sensor, with_resources
from dagster._check import CheckError
from dagster._config.pythonic_config import ConfigurableResource
from dagster._core.definitions.cacheable_assets import AssetsDefinitionCacheableData, CacheableAssetsDefinition
from dagster._core.definitions.decorators.job_decorator import job
from dagster._core.definitions.executor_definition import executor
from dagster._core.definitions.job_definition import JobDefinition
from dagster._core.definitions.logger_definition import logger
from dagster._core.definitions.repository_definition import PendingRepositoryDefinition, RepositoryDefinition
from dagster._core.definitions.sensor_definition import SensorDefinition
from dagster._core.definitions.time_window_partitions import DailyPartitionsDefinition, HourlyPartitionsDefinition
from dagster._core.errors import DagsterInvariantViolationError
from dagster._core.executor.base import Executor
from dagster._core.storage.io_manager import IOManagerDefinition
from dagster._core.storage.mem_io_manager import InMemoryIOManager
from dagster._core.test_utils import instance_for_test

def get_all_assets_from_defs(defs: Definitions):
    if False:
        return 10
    repo = resolve_pending_repo_if_required(defs)
    return list(repo.assets_defs_by_key.values())

def resolve_pending_repo_if_required(definitions: Definitions) -> RepositoryDefinition:
    if False:
        print('Hello World!')
    repo_or_caching_repo = definitions.get_inner_repository_for_loading_process()
    return repo_or_caching_repo.compute_repository_definition() if isinstance(repo_or_caching_repo, PendingRepositoryDefinition) else repo_or_caching_repo

def test_basic_asset():
    if False:
        i = 10
        return i + 15
    assert Definitions

    @asset
    def an_asset():
        if False:
            for i in range(10):
                print('nop')
        pass
    defs = Definitions(assets=[an_asset])
    all_assets = get_all_assets_from_defs(defs)
    assert len(all_assets) == 1
    assert all_assets[0].key.to_user_string() == 'an_asset'

def test_basic_asset_job_definition():
    if False:
        i = 10
        return i + 15

    @asset
    def an_asset():
        if False:
            return 10
        pass
    defs = Definitions(assets=[an_asset], jobs=[define_asset_job(name='an_asset_job')])
    assert isinstance(defs.get_job_def('an_asset_job'), JobDefinition)

def test_vanilla_job_definition():
    if False:
        while True:
            i = 10

    @op
    def an_op():
        if False:
            for i in range(10):
                print('nop')
        pass

    @job
    def a_job():
        if False:
            return 10
        pass
    defs = Definitions(jobs=[a_job])
    assert isinstance(defs.get_job_def('a_job'), JobDefinition)

def test_basic_schedule_definition():
    if False:
        return 10

    @asset
    def an_asset():
        if False:
            for i in range(10):
                print('nop')
        pass
    defs = Definitions(assets=[an_asset], schedules=[ScheduleDefinition(name='daily_an_asset_schedule', job=define_asset_job(name='an_asset_job'), cron_schedule='@daily')])
    assert defs.get_schedule_def('daily_an_asset_schedule')

def test_basic_sensor_definition():
    if False:
        for i in range(10):
            print('nop')

    @asset
    def an_asset():
        if False:
            i = 10
            return i + 15
        pass
    an_asset_job = define_asset_job(name='an_asset_job')

    @sensor(name='an_asset_sensor', job=an_asset_job)
    def a_sensor():
        if False:
            i = 10
            return i + 15
        raise NotImplementedError()
    defs = Definitions(assets=[an_asset], sensors=[a_sensor])
    assert defs.get_sensor_def('an_asset_sensor')

def test_with_resource_binding():
    if False:
        while True:
            i = 10
    executed = {}

    @asset(required_resource_keys={'foo'})
    def requires_foo(context):
        if False:
            print('Hello World!')
        assert context.resources.foo == 'wrapped'
        executed['yes'] = True
    defs = Definitions(assets=[requires_foo], resources={'foo': ResourceDefinition.hardcoded_resource('wrapped')})
    repo = resolve_pending_repo_if_required(defs)
    assert len(repo.get_top_level_resources()) == 1
    assert 'foo' in repo.get_top_level_resources()
    asset_job = repo.get_all_jobs()[0]
    asset_job.execute_in_process()
    assert executed['yes']

def test_nested_resources() -> None:
    if False:
        print('Hello World!')

    class MyInnerResource(ConfigurableResource):
        a_str: str

    class MyOuterResource(ConfigurableResource):
        inner: MyInnerResource
    inner = MyInnerResource(a_str='wrapped')
    defs = Definitions(resources={'foo': MyOuterResource(inner=inner)})
    repo = resolve_pending_repo_if_required(defs)
    assert len(repo.get_top_level_resources()) == 1
    assert 'foo' in repo.get_top_level_resources()

def test_resource_coercion():
    if False:
        print('Hello World!')
    executed = {}

    @asset(required_resource_keys={'foo'})
    def requires_foo(context):
        if False:
            return 10
        assert context.resources.foo == 'object-to-coerce'
        executed['yes'] = True
    defs = Definitions(assets=[requires_foo], resources={'foo': 'object-to-coerce'})
    repo = resolve_pending_repo_if_required(defs)
    assert len(repo.get_top_level_resources()) == 1
    assert 'foo' in repo.get_top_level_resources()
    asset_job = repo.get_all_jobs()[0]
    asset_job.execute_in_process()
    assert executed['yes']

def test_source_asset():
    if False:
        i = 10
        return i + 15
    defs = Definitions(assets=[SourceAsset('a-source-asset')])
    repo = resolve_pending_repo_if_required(defs)
    all_assets = list(repo.source_assets_by_key.values())
    assert len(all_assets) == 1
    assert all_assets[0].key.to_user_string() == 'a-source-asset'

def test_pending_repo():
    if False:
        for i in range(10):
            print('nop')

    class MyCacheableAssetsDefinition(CacheableAssetsDefinition):

        def compute_cacheable_data(self):
            if False:
                return 10
            return [AssetsDefinitionCacheableData(keys_by_input_name={}, keys_by_output_name={'result': AssetKey(self.unique_id)})]

        def build_definitions(self, data):
            if False:
                return 10

            @op
            def my_op():
                if False:
                    while True:
                        i = 10
                return 1
            return [AssetsDefinition.from_op(my_op, keys_by_input_name=cd.keys_by_input_name, keys_by_output_name=cd.keys_by_output_name) for cd in data]

    @repository
    def a_pending_repo():
        if False:
            print('Hello World!')
        return [MyCacheableAssetsDefinition('foobar')]
    assert isinstance(a_pending_repo, PendingRepositoryDefinition)
    assert isinstance(a_pending_repo.compute_repository_definition(), RepositoryDefinition)
    defs = Definitions(assets=[MyCacheableAssetsDefinition('foobar')])
    all_assets = get_all_assets_from_defs(defs)
    assert len(all_assets) == 1
    assert all_assets[0].key.to_user_string() == 'foobar'
    assert isinstance(defs.get_implicit_global_asset_job_def(), JobDefinition)

def test_asset_loading():
    if False:
        print('Hello World!')

    @asset
    def one():
        if False:
            return 10
        return 1

    @asset
    def two():
        if False:
            print('Hello World!')
        return 2
    with instance_for_test() as instance:
        defs = Definitions(assets=[one, two])
        materialize(assets=[one, two], instance=instance)
        assert defs.load_asset_value('one', instance=instance) == 1
        assert defs.load_asset_value('two', instance=instance) == 2
        value_loader = defs.get_asset_value_loader(instance)
        assert value_loader.load_asset_value('one') == 1
        assert value_loader.load_asset_value('two') == 2

def test_io_manager_coercion():
    if False:
        return 10

    @asset(io_manager_key='mem_io_manager')
    def one():
        if False:
            print('Hello World!')
        return 1
    defs = Definitions(assets=[one], resources={'mem_io_manager': InMemoryIOManager()})
    asset_job = defs.get_implicit_global_asset_job_def()
    assert isinstance(asset_job.resource_defs['mem_io_manager'], IOManagerDefinition)
    result = asset_job.execute_in_process()
    assert result.output_for_node('one') == 1

def test_bad_executor():
    if False:
        while True:
            i = 10
    with pytest.raises(CheckError):
        Definitions(executor='not an executor')

def test_custom_executor_in_definitions():
    if False:
        print('Hello World!')

    @executor
    def an_executor(_):
        if False:
            print('Hello World!')
        raise Exception('not executed')

    @asset
    def one():
        if False:
            for i in range(10):
                print('nop')
        return 1
    defs = Definitions(assets=[one], executor=an_executor)
    asset_job = defs.get_implicit_global_asset_job_def()
    assert asset_job.executor_def is an_executor

def test_custom_loggers_in_definitions():
    if False:
        i = 10
        return i + 15

    @logger
    def a_logger(_):
        if False:
            return 10
        raise Exception('not executed')

    @asset
    def one():
        if False:
            return 10
        return 1
    defs = Definitions(assets=[one], loggers={'custom_logger': a_logger})
    asset_job = defs.get_implicit_global_asset_job_def()
    loggers = asset_job.loggers
    assert len(loggers) == 1
    assert 'custom_logger' in loggers
    assert loggers['custom_logger'] is a_logger

def test_bad_logger_key():
    if False:
        while True:
            i = 10

    @logger
    def a_logger(_):
        if False:
            return 10
        raise Exception('not executed')
    with pytest.raises(CheckError):
        Definitions(loggers={1: a_logger})

def test_bad_logger_value():
    if False:
        i = 10
        return i + 15
    with pytest.raises(CheckError):
        Definitions(loggers={'not_a_logger': 'not_a_logger'})

def test_kitchen_sink_on_create_helper_and_definitions():
    if False:
        print('Hello World!')

    @asset(required_resource_keys={'a_resource_key'})
    def an_asset():
        if False:
            return 10
        pass

    @asset
    def another_asset():
        if False:
            while True:
                i = 10
        pass
    another_asset_job = define_asset_job(name='another_asset_job', selection='another_asset')

    @op
    def an_op():
        if False:
            print('Hello World!')
        pass

    @job
    def a_job():
        if False:
            return 10
        an_op()

    @job
    def sensor_target():
        if False:
            return 10
        an_op()

    @job
    def schedule_target():
        if False:
            return 10
        an_op()
    a_schedule = ScheduleDefinition(name='a_schedule', job=schedule_target, cron_schedule='@daily')

    @sensor(job=sensor_target)
    def a_sensor(_):
        if False:
            i = 10
            return i + 15
        raise Exception('not called')

    @executor
    def an_executor(_):
        if False:
            return 10
        raise Exception('not executed')

    @logger
    def a_logger(_):
        if False:
            i = 10
            return i + 15
        raise Exception('not executed')
    repo = create_repository_using_definitions_args(name='foobar', assets=[an_asset, another_asset], jobs=[a_job, another_asset_job], schedules=[a_schedule], sensors=[a_sensor], resources={'a_resource_key': 'the resource'}, executor=an_executor, loggers={'logger_key': a_logger})
    assert isinstance(repo, RepositoryDefinition)
    assert repo.name == 'foobar'
    assert isinstance(repo.get_job('a_job'), JobDefinition)
    assert repo.get_job('a_job').executor_def is an_executor
    assert repo.get_job('a_job').loggers == {'logger_key': a_logger}
    assert isinstance(repo.get_implicit_global_asset_job_def(), JobDefinition)
    assert repo.get_implicit_global_asset_job_def().executor_def is an_executor
    assert repo.get_implicit_global_asset_job_def().loggers == {'logger_key': a_logger}
    assert isinstance(repo.get_job('another_asset_job'), JobDefinition)
    assert repo.get_job('another_asset_job').executor_def is an_executor
    assert repo.get_job('another_asset_job').loggers == {'logger_key': a_logger}
    assert isinstance(repo.get_job('sensor_target'), JobDefinition)
    assert repo.get_job('sensor_target').executor_def is an_executor
    assert repo.get_job('sensor_target').loggers == {'logger_key': a_logger}
    assert isinstance(repo.get_job('schedule_target'), JobDefinition)
    assert repo.get_job('schedule_target').executor_def is an_executor
    assert repo.get_job('schedule_target').loggers == {'logger_key': a_logger}
    assert isinstance(repo.get_schedule_def('a_schedule'), ScheduleDefinition)
    assert isinstance(repo.get_sensor_def('a_sensor'), SensorDefinition)
    defs = Definitions(assets=[an_asset, another_asset], jobs=[a_job, another_asset_job], schedules=[a_schedule], sensors=[a_sensor], resources={'a_resource_key': 'the resource'}, executor=an_executor, loggers={'logger_key': a_logger})
    assert isinstance(defs.get_job_def('a_job'), JobDefinition)
    assert defs.get_job_def('a_job').executor_def is an_executor
    assert defs.get_job_def('a_job').loggers == {'logger_key': a_logger}
    assert isinstance(defs.get_implicit_global_asset_job_def(), JobDefinition)
    assert defs.get_implicit_global_asset_job_def().executor_def is an_executor
    assert defs.get_implicit_global_asset_job_def().loggers == {'logger_key': a_logger}
    assert isinstance(defs.get_job_def('another_asset_job'), JobDefinition)
    assert defs.get_job_def('another_asset_job').executor_def is an_executor
    assert defs.get_job_def('another_asset_job').loggers == {'logger_key': a_logger}
    assert isinstance(defs.get_job_def('sensor_target'), JobDefinition)
    assert defs.get_job_def('sensor_target').executor_def is an_executor
    assert defs.get_job_def('sensor_target').loggers == {'logger_key': a_logger}
    assert isinstance(defs.get_job_def('schedule_target'), JobDefinition)
    assert defs.get_job_def('schedule_target').executor_def is an_executor
    assert defs.get_job_def('schedule_target').loggers == {'logger_key': a_logger}
    assert isinstance(defs.get_schedule_def('a_schedule'), ScheduleDefinition)
    assert isinstance(defs.get_sensor_def('a_sensor'), SensorDefinition)

def test_with_resources_override():
    if False:
        for i in range(10):
            print('nop')
    executed = {}

    @asset(required_resource_keys={'a_resource'})
    def asset_one(context):
        if False:
            i = 10
            return i + 15
        executed['asset_one'] = True
        assert context.resources.a_resource == 'passed-through-with-resources'

    @asset(required_resource_keys={'b_resource'})
    def asset_two(context):
        if False:
            return 10
        executed['asset_two'] = True
        assert context.resources.b_resource == 'passed-through-definitions'
    defs = Definitions(assets=[*with_resources([asset_one], {'a_resource': ResourceDefinition.hardcoded_resource('passed-through-with-resources')}), asset_two], resources={'b_resource': 'passed-through-definitions'})
    defs.get_implicit_global_asset_job_def().execute_in_process()
    assert executed['asset_one']
    assert executed['asset_two']

def test_implicit_global_job():
    if False:
        while True:
            i = 10

    @asset
    def asset_one():
        if False:
            print('Hello World!')
        pass
    defs = Definitions(assets=[asset_one])
    assert defs.has_implicit_global_asset_job_def()
    assert len(defs.get_all_job_defs()) == 1

def test_implicit_global_job_with_job_defined():
    if False:
        while True:
            i = 10

    @asset
    def asset_one():
        if False:
            while True:
                i = 10
        pass
    defs = Definitions(assets=[asset_one], jobs=[define_asset_job('all_assets_job', selection='*')])
    assert defs.has_implicit_global_asset_job_def()
    assert defs.get_job_def('all_assets_job')
    assert defs.get_job_def('all_assets_job') is not defs.get_implicit_global_asset_job_def()
    assert len(defs.get_all_job_defs()) == 2

def test_implicit_global_job_with_partitioned_asset():
    if False:
        return 10

    @asset(partitions_def=DailyPartitionsDefinition(start_date='2022-01-01'))
    def daily_partition_asset(context: AssetExecutionContext):
        if False:
            i = 10
            return i + 15
        return context.partition_key

    @asset(partitions_def=HourlyPartitionsDefinition(start_date='2022-02-02-10:00'))
    def hourly_partition_asset(context: AssetExecutionContext):
        if False:
            i = 10
            return i + 15
        return context.partition_key

    @asset
    def unpartitioned_asset():
        if False:
            i = 10
            return i + 15
        pass
    defs = Definitions(assets=[daily_partition_asset, unpartitioned_asset, hourly_partition_asset])
    assert len(defs.get_all_job_defs()) == 2
    assert defs.get_implicit_job_def_for_assets([AssetKey('daily_partition_asset'), AssetKey('unpartitioned_asset')])
    assert defs.get_implicit_job_def_for_assets([AssetKey('hourly_partition_asset'), AssetKey('unpartitioned_asset')])
    with pytest.raises(DagsterInvariantViolationError):
        defs.get_implicit_global_asset_job_def()

def test_implicit_job_with_source_assets():
    if False:
        i = 10
        return i + 15
    source_asset = SourceAsset('source_asset')

    @asset
    def downstream_of_source(source_asset):
        if False:
            print('Hello World!')
        raise Exception('not executed')
    defs = Definitions(assets=[source_asset, downstream_of_source])
    assert defs.get_all_job_defs()
    assert len(defs.get_all_job_defs()) == 1
    assert defs.get_implicit_job_def_for_assets(asset_keys=[AssetKey('downstream_of_source')])
    assert defs.has_implicit_global_asset_job_def()
    assert defs.get_implicit_global_asset_job_def()

def test_unresolved_partitioned_asset_schedule():
    if False:
        print('Hello World!')
    partitions_def = DailyPartitionsDefinition(start_date='2020-01-01')

    @asset(partitions_def=partitions_def)
    def asset1():
        if False:
            for i in range(10):
                print('nop')
        ...
    job1 = define_asset_job('job1')
    schedule1 = build_schedule_from_partitioned_job(job1)
    defs_with_explicit_job = Definitions(jobs=[job1], schedules=[schedule1], assets=[asset1])
    assert defs_with_explicit_job.get_job_def('job1').name == 'job1'
    assert defs_with_explicit_job.get_job_def('job1').partitions_def == partitions_def
    assert defs_with_explicit_job.get_schedule_def('job1_schedule').cron_schedule == '0 0 * * *'
    defs_with_implicit_job = Definitions(schedules=[schedule1], assets=[asset1])
    assert defs_with_implicit_job.get_job_def('job1').name == 'job1'
    assert defs_with_implicit_job.get_job_def('job1').partitions_def == partitions_def
    assert defs_with_implicit_job.get_schedule_def('job1_schedule').cron_schedule == '0 0 * * *'

def test_bare_executor():
    if False:
        i = 10
        return i + 15

    @asset
    def an_asset():
        if False:
            return 10
        ...

    class DummyExecutor(Executor):

        def execute(self, plan_context, execution_plan):
            if False:
                for i in range(10):
                    print('nop')
            ...

        @property
        def retries(self):
            if False:
                for i in range(10):
                    print('nop')
            ...
    executor_inst = DummyExecutor()
    defs = Definitions(assets=[an_asset], executor=executor_inst)
    job = defs.get_implicit_global_asset_job_def()
    assert isinstance(job, JobDefinition)
    assert job.executor_def.executor_creation_fn(None) is executor_inst

def test_assets_with_io_manager():
    if False:
        while True:
            i = 10

    @asset
    def single_asset():
        if False:
            print('Hello World!')
        pass
    defs = Definitions(assets=[single_asset], resources={'io_manager': mem_io_manager})
    asset_group_underlying_job = defs.get_all_job_defs()[0]
    assert asset_group_underlying_job.resource_defs['io_manager'] == mem_io_manager

def test_asset_missing_resources():
    if False:
        print('Hello World!')

    @asset(required_resource_keys={'foo'})
    def asset_foo(context):
        if False:
            return 10
        return context.resources.foo
    with pytest.raises(DagsterInvalidDefinitionError, match="resource with key 'foo' required by op 'asset_foo' was not provided."):
        Definitions(assets=[asset_foo])
    source_asset_io_req = SourceAsset(key=AssetKey('foo'), io_manager_key='foo')
    with pytest.raises(DagsterInvalidDefinitionError, match=re.escape('io manager with key \'foo\' required by SourceAsset with key ["foo"] was not provided')):
        Definitions(assets=[source_asset_io_req])

def test_assets_with_executor():
    if False:
        while True:
            i = 10

    @asset
    def the_asset():
        if False:
            i = 10
            return i + 15
        pass
    defs = Definitions(assets=[the_asset], executor=in_process_executor)
    asset_group_underlying_job = defs.get_all_job_defs()[0]
    assert asset_group_underlying_job.executor_def == in_process_executor

def test_asset_missing_io_manager():
    if False:
        print('Hello World!')

    @asset(io_manager_key='blah')
    def asset_foo():
        if False:
            print('Hello World!')
        pass
    with pytest.raises(DagsterInvalidDefinitionError, match="io manager with key 'blah' required by output 'result' of op 'asset_foo'' was not provided."):
        Definitions(assets=[asset_foo])

def test_resource_defs_on_asset():
    if False:
        i = 10
        return i + 15
    the_resource = ResourceDefinition.hardcoded_resource('blah')

    @asset(required_resource_keys={'bar'}, resource_defs={'foo': the_resource})
    def the_asset():
        if False:
            i = 10
            return i + 15
        pass

    @asset(resource_defs={'foo': the_resource})
    def other_asset():
        if False:
            print('Hello World!')
        pass
    defs = Definitions([the_asset, other_asset], resources={'bar': the_resource})
    the_job = defs.get_all_job_defs()[0]
    assert the_job.execute_in_process().success

def test_conflicting_asset_resource_defs():
    if False:
        while True:
            i = 10
    the_resource = ResourceDefinition.hardcoded_resource('blah')
    other_resource = ResourceDefinition.hardcoded_resource('baz')

    @asset(resource_defs={'foo': the_resource})
    def the_asset():
        if False:
            while True:
                i = 10
        pass

    @asset(resource_defs={'foo': other_resource})
    def other_asset():
        if False:
            for i in range(10):
                print('nop')
        pass
    with pytest.raises(DagsterInvalidDefinitionError, match="Conflicting versions of resource with key 'foo' were provided to different assets. When constructing a job, all resource definitions provided to assets must match by reference equality for a given key."):
        Definitions([the_asset, other_asset])

def test_graph_backed_asset_resources():
    if False:
        return 10

    @op(required_resource_keys={'foo'})
    def the_op():
        if False:
            return 10
        pass

    @graph
    def basic():
        if False:
            i = 10
            return i + 15
        return the_op()
    the_resource = ResourceDefinition.hardcoded_resource('blah')
    other_resource = ResourceDefinition.hardcoded_resource('baz')
    the_asset = AssetsDefinition.from_graph(graph_def=basic, keys_by_input_name={}, keys_by_output_name={'result': AssetKey('the_asset')}, resource_defs={'foo': the_resource})
    Definitions([the_asset])
    other_asset = AssetsDefinition.from_graph(keys_by_input_name={}, keys_by_output_name={'result': AssetKey('other_asset')}, graph_def=basic, resource_defs={'foo': other_resource})
    with pytest.raises(DagsterInvalidDefinitionError, match="Conflicting versions of resource with key 'foo' were provided to different assets. When constructing a job, all resource definitions provided to assets must match by reference equality for a given key."):
        Definitions([the_asset, other_asset])

def test_job_with_reserved_name():
    if False:
        for i in range(10):
            print('nop')

    @graph
    def the_graph():
        if False:
            i = 10
            return i + 15
        pass
    the_job = the_graph.to_job(name='__ASSET_JOB')
    with pytest.raises(DagsterInvalidDefinitionError, match='Attempted to provide job called __ASSET_JOB to repository, which is a reserved name.'):
        Definitions(jobs=[the_job])

def test_asset_cycle():
    if False:
        return 10
    from toposort import CircularDependencyError

    @asset
    def a(s, c):
        if False:
            while True:
                i = 10
        return s + c

    @asset
    def b(a):
        if False:
            for i in range(10):
                print('nop')
        return a + 1

    @asset
    def c(b):
        if False:
            print('Hello World!')
        return b + 1
    s = SourceAsset(key='s')
    with pytest.raises(CircularDependencyError):
        Definitions(assets=[a, b, c, s])