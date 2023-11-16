from collections import defaultdict
from typing import Sequence
import pytest
from dagster import AssetKey, AssetsDefinition, DagsterInvalidDefinitionError, DailyPartitionsDefinition, GraphDefinition, IOManager, JobDefinition, OpDefinition, ResourceDefinition, SensorDefinition, SourceAsset, asset, build_schedule_from_partitioned_job, define_asset_job, executor, graph, in_process_executor, io_manager, job, logger, op, repository, resource, schedule, sensor
from dagster._check import CheckError
from dagster._core.definitions.executor_definition import multi_or_in_process_executor
from dagster._core.definitions.partition import PartitionedConfig, StaticPartitionsDefinition
from dagster._core.errors import DagsterInvalidSubsetError
from dagster._loggers import default_loggers

def create_single_node_job(name, called):
    if False:
        print('Hello World!')
    called[name] = called[name] + 1
    return JobDefinition(graph_def=GraphDefinition(name=name, node_defs=[OpDefinition(name=name + '_op', ins={}, outs={}, compute_fn=lambda *_args, **_kwargs: None)]))

def test_repo_lazy_definition():
    if False:
        for i in range(10):
            print('nop')
    called = defaultdict(int)

    @repository
    def lazy_repo():
        if False:
            while True:
                i = 10
        return {'jobs': {'foo': lambda : create_single_node_job('foo', called), 'bar': lambda : create_single_node_job('bar', called)}}
    foo_job = lazy_repo.get_job('foo')
    assert isinstance(foo_job, JobDefinition)
    assert foo_job.name == 'foo'
    assert 'foo' in called
    assert called['foo'] == 1
    assert 'bar' not in called
    bar_job = lazy_repo.get_job('bar')
    assert isinstance(bar_job, JobDefinition)
    assert bar_job.name == 'bar'
    assert 'foo' in called
    assert called['foo'] == 1
    assert 'bar' in called
    assert called['bar'] == 1
    foo_job = lazy_repo.get_job('foo')
    assert isinstance(foo_job, JobDefinition)
    assert foo_job.name == 'foo'
    assert 'foo' in called
    assert called['foo'] == 1
    jobs = lazy_repo.get_all_jobs()
    assert set(['foo', 'bar']) == {job.name for job in jobs}

def test_dupe_op_repo_definition():
    if False:
        while True:
            i = 10

    @op(name='same')
    def noop():
        if False:
            return 10
        pass

    @op(name='same')
    def noop2():
        if False:
            print('Hello World!')
        pass

    @repository
    def error_repo():
        if False:
            for i in range(10):
                print('nop')
        return {'jobs': {'first': lambda : JobDefinition(graph_def=GraphDefinition(name='first', node_defs=[noop])), 'second': lambda : JobDefinition(graph_def=GraphDefinition(name='second', node_defs=[noop2]))}}
    with pytest.raises(DagsterInvalidDefinitionError, match="Conflicting definitions found in repository with name 'same'. Op/Graph definition names must be unique within a repository."):
        error_repo.get_all_jobs()

def test_non_lazy_job_dict():
    if False:
        for i in range(10):
            print('nop')
    called = defaultdict(int)

    @repository
    def some_repo():
        if False:
            i = 10
            return i + 15
        return [create_single_node_job('foo', called), create_single_node_job('bar', called)]
    assert some_repo.get_job('foo').name == 'foo'
    assert some_repo.get_job('bar').name == 'bar'

def test_conflict():
    if False:
        while True:
            i = 10
    called = defaultdict(int)
    with pytest.raises(Exception, match="Duplicate job definition found for job 'foo'"):

        @repository
        def _some_repo():
            if False:
                while True:
                    i = 10
            return [create_single_node_job('foo', called), create_single_node_job('foo', called)]

def test_key_mismatch():
    if False:
        i = 10
        return i + 15
    called = defaultdict(int)

    @repository
    def some_repo():
        if False:
            return 10
        return {'jobs': {'foo': lambda : create_single_node_job('bar', called)}}
    with pytest.raises(Exception, match='name in JobDefinition does not match'):
        some_repo.get_job('foo')

def test_non_job_in_jobs():
    if False:
        print('Hello World!')
    with pytest.raises(DagsterInvalidDefinitionError, match='all elements of list must be of type'):

        @repository
        def _some_repo():
            if False:
                i = 10
                return i + 15
            return ['not-a-job']

def test_bad_schedule():
    if False:
        for i in range(10):
            print('nop')

    @schedule(cron_schedule='* * * * *', job_name='foo')
    def daily_foo(context):
        if False:
            i = 10
            return i + 15
        return {}
    with pytest.raises(Exception, match='targets job "foo" which was not found in this repository'):

        @repository
        def _some_repo():
            if False:
                return 10
            return [daily_foo]

def test_bad_sensor():
    if False:
        i = 10
        return i + 15

    @sensor(job_name='foo')
    def foo_sensor(_):
        if False:
            i = 10
            return i + 15
        return {}
    with pytest.raises(DagsterInvalidDefinitionError, match='targets job "foo" which was not found in this repository'):

        @repository
        def _some_repo():
            if False:
                print('Hello World!')
            return [foo_sensor]

def test_direct_schedule_target():
    if False:
        print('Hello World!')

    @op
    def wow():
        if False:
            print('Hello World!')
        return 'wow'

    @graph
    def wonder():
        if False:
            i = 10
            return i + 15
        wow()

    @schedule(cron_schedule='* * * * *', job=wonder)
    def direct_schedule():
        if False:
            print('Hello World!')
        return {}

    @repository
    def test():
        if False:
            i = 10
            return i + 15
        return [direct_schedule]
    assert test

def test_direct_schedule_unresolved_target():
    if False:
        i = 10
        return i + 15
    unresolved_job = define_asset_job('unresolved_job', selection='foo')

    @asset
    def foo():
        if False:
            for i in range(10):
                print('nop')
        return None

    @schedule(cron_schedule='* * * * *', job=unresolved_job)
    def direct_schedule():
        if False:
            while True:
                i = 10
        return {}

    @repository
    def test():
        if False:
            i = 10
            return i + 15
        return [direct_schedule, foo]
    assert isinstance(test.get_job('unresolved_job'), JobDefinition)

def test_direct_sensor_target():
    if False:
        i = 10
        return i + 15

    @op
    def wow():
        if False:
            i = 10
            return i + 15
        return 'wow'

    @graph
    def wonder():
        if False:
            while True:
                i = 10
        wow()

    @sensor(job=wonder)
    def direct_sensor(_):
        if False:
            while True:
                i = 10
        return {}

    @repository
    def test():
        if False:
            i = 10
            return i + 15
        return [direct_sensor]
    assert test

def test_direct_sensor_unresolved_target():
    if False:
        while True:
            i = 10
    unresolved_job = define_asset_job('unresolved_job', selection='foo')

    @asset
    def foo():
        if False:
            while True:
                i = 10
        return None

    @sensor(job=unresolved_job)
    def direct_sensor(_):
        if False:
            i = 10
            return i + 15
        return {}

    @repository
    def test():
        if False:
            print('Hello World!')
        return [direct_sensor, foo]
    assert isinstance(test.get_job('unresolved_job'), JobDefinition)

def test_target_dupe_job():
    if False:
        while True:
            i = 10

    @op
    def wow():
        if False:
            for i in range(10):
                print('nop')
        return 'wow'

    @graph
    def wonder():
        if False:
            print('Hello World!')
        wow()
    w_job = wonder.to_job()

    @sensor(job=w_job)
    def direct_sensor(_):
        if False:
            i = 10
            return i + 15
        return {}

    @repository
    def test():
        if False:
            print('Hello World!')
        return [direct_sensor, w_job]
    assert test

def test_target_dupe_unresolved():
    if False:
        while True:
            i = 10
    unresolved_job = define_asset_job('unresolved_job', selection='foo')

    @asset
    def foo():
        if False:
            while True:
                i = 10
        return None

    @sensor(job=unresolved_job)
    def direct_sensor(_):
        if False:
            print('Hello World!')
        return {}

    @repository
    def test():
        if False:
            while True:
                i = 10
        return [foo, direct_sensor, unresolved_job]
    assert isinstance(test.get_job('unresolved_job'), JobDefinition)

def test_bare_graph():
    if False:
        while True:
            i = 10

    @op
    def ok():
        if False:
            for i in range(10):
                print('nop')
        return 'sure'

    @graph
    def bare():
        if False:
            i = 10
            return i + 15
        ok()

    @repository
    def test():
        if False:
            return 10
        return [bare]
    assert test.get_job('bare')
    assert test.get_job('bare')

def test_unresolved_job():
    if False:
        print('Hello World!')
    unresolved_job = define_asset_job('unresolved_job', selection='foo')

    @asset
    def foo():
        if False:
            for i in range(10):
                print('nop')
        return None

    @repository
    def test():
        if False:
            return 10
        return [foo, unresolved_job]
    assert isinstance(test.get_job('unresolved_job'), JobDefinition)
    assert isinstance(test.get_job('unresolved_job'), JobDefinition)

def test_bare_graph_with_resources():
    if False:
        while True:
            i = 10

    @op(required_resource_keys={'stuff'})
    def needy(context):
        if False:
            i = 10
            return i + 15
        return context.resources.stuff

    @graph
    def bare():
        if False:
            while True:
                i = 10
        needy()
    with pytest.raises(DagsterInvalidDefinitionError, match="resource with key 'stuff' required by op 'needy' was not provided"):

        @repository
        def _test():
            if False:
                i = 10
                return i + 15
            return [bare]

def test_sensor_no_job_name():
    if False:
        while True:
            i = 10
    foo_system_sensor = SensorDefinition(name='foo', evaluation_fn=lambda x: x)

    @repository
    def foo_repo():
        if False:
            print('Hello World!')
        return [foo_system_sensor]
    assert foo_repo.has_sensor_def('foo')

def test_job_with_partitions():
    if False:
        i = 10
        return i + 15

    @op
    def ok():
        if False:
            return 10
        return 'sure'

    @graph
    def bare():
        if False:
            for i in range(10):
                print('nop')
        ok()

    @repository
    def test():
        if False:
            return 10
        return [bare.to_job(resource_defs={}, config=PartitionedConfig(partitions_def=StaticPartitionsDefinition(['abc']), run_config_for_partition_key_fn=lambda _: {}))]
    assert test.has_job('bare')
    assert test.get_job('bare').partitions_def
    assert test.has_job('bare')
    assert test.get_job('bare').partitions_def

def test_dupe_graph_defs():
    if False:
        return 10

    @op
    def noop():
        if False:
            return 10
        pass

    @job(name='foo')
    def job_foo():
        if False:
            return 10
        noop()

    @graph(name='foo')
    def graph_foo():
        if False:
            for i in range(10):
                print('nop')
        noop()
    with pytest.raises(DagsterInvalidDefinitionError, match="Duplicate job definition found for job 'foo'"):

        @repository
        def _job_collide():
            if False:
                for i in range(10):
                    print('nop')
            return [graph_foo, job_foo]

    def get_collision_repo():
        if False:
            return 10

        @repository
        def graph_collide():
            if False:
                while True:
                    i = 10
            return [graph_foo.to_job(name='bar'), job_foo]
        return graph_collide
    with pytest.raises(DagsterInvalidDefinitionError, match='Op/Graph definition names must be unique within a repository'):
        get_collision_repo().get_all_jobs()
    with pytest.raises(DagsterInvalidDefinitionError, match='Op/Graph definition names must be unique within a repository'):
        get_collision_repo().get_all_jobs()

def test_dupe_unresolved_job_defs():
    if False:
        print('Hello World!')
    unresolved_job = define_asset_job('bar', selection='foo')

    @asset
    def foo():
        if False:
            for i in range(10):
                print('nop')
        return None

    @op
    def the_op():
        if False:
            print('Hello World!')
        pass

    @graph
    def graph_bar():
        if False:
            while True:
                i = 10
        the_op()
    bar = graph_bar.to_job(name='bar')
    with pytest.raises(DagsterInvalidDefinitionError, match="Duplicate job definition found for job 'bar'"):

        @repository
        def _pipe_collide():
            if False:
                return 10
            return [foo, unresolved_job, bar]

    def get_collision_repo():
        if False:
            while True:
                i = 10

        @repository
        def graph_collide():
            if False:
                i = 10
                return i + 15
            return [foo, graph_bar.to_job(name='bar'), unresolved_job]
        return graph_collide
    with pytest.raises(DagsterInvalidDefinitionError, match="Duplicate definition found for unresolved job 'bar'"):
        get_collision_repo().get_all_jobs()
    with pytest.raises(DagsterInvalidDefinitionError, match="Duplicate definition found for unresolved job 'bar'"):
        get_collision_repo().get_all_jobs()

def test_job_validation():
    if False:
        while True:
            i = 10
    with pytest.raises(DagsterInvalidDefinitionError, match='Object mapped to my_job is not an instance of JobDefinition or GraphDefinition.'):

        @repository
        def _my_repo():
            if False:
                return 10
            return {'jobs': {'my_job': 'blah'}}

def test_dict_jobs():
    if False:
        return 10

    @graph
    def my_graph():
        if False:
            while True:
                i = 10
        pass

    @repository
    def jobs():
        if False:
            print('Hello World!')
        return {'jobs': {'my_graph': my_graph, 'other_graph': my_graph.to_job(name='other_graph'), 'tbd': define_asset_job('tbd', selection='*')}}
    assert jobs.get_job('my_graph')
    assert jobs.get_job('other_graph')
    assert jobs.has_job('my_graph')
    assert jobs.get_job('my_graph')
    assert jobs.get_job('other_graph')
    assert jobs.has_job('tbd')
    assert jobs.get_job('tbd')

def test_lazy_jobs():
    if False:
        while True:
            i = 10

    @graph
    def my_graph():
        if False:
            print('Hello World!')
        pass

    @repository
    def jobs():
        if False:
            return 10
        return {'jobs': {'my_graph': my_graph, 'my_job': lambda : my_graph.to_job(name='my_job'), 'other_job': lambda : my_graph.to_job(name='other_job')}}
    assert jobs.get_job('my_graph')
    assert jobs.get_job('my_job')
    assert jobs.get_job('other_job')
    assert jobs.has_job('my_graph')
    assert jobs.get_job('my_job')
    assert jobs.get_job('other_job')

def test_lazy_graph():
    if False:
        while True:
            i = 10

    @graph
    def my_graph():
        if False:
            i = 10
            return i + 15
        pass

    @repository
    def jobs():
        if False:
            print('Hello World!')
        return {'jobs': {'my_graph': lambda : my_graph}}
    with pytest.raises(CheckError, match='Invariant failed. Description: Bad constructor for job my_graph: must return JobDefinition'):
        assert jobs.get_job('my_graph')

def test_list_dupe_graph():
    if False:
        i = 10
        return i + 15

    @graph
    def foo():
        if False:
            return 10
        pass
    with pytest.raises(DagsterInvalidDefinitionError, match="Duplicate job definition found for graph 'foo'"):

        @repository
        def _jobs():
            if False:
                i = 10
                return i + 15
            return [foo.to_job(name='foo'), foo]

def test_bad_coerce():
    if False:
        return 10

    @op(required_resource_keys={'x'})
    def foo():
        if False:
            return 10
        pass

    @graph
    def bar():
        if False:
            print('Hello World!')
        foo()
    with pytest.raises(DagsterInvalidDefinitionError, match="resource with key 'x' required by op 'foo' was not provided"):

        @repository
        def _fails():
            if False:
                for i in range(10):
                    print('nop')
            return {'jobs': {'bar': bar}}

def test_bad_resolve():
    if False:
        print('Hello World!')
    with pytest.raises(DagsterInvalidSubsetError, match="AssetKey\\(s\\) {AssetKey\\(\\['foo'\\]\\)} were selected"):

        @repository
        def _fails():
            if False:
                for i in range(10):
                    print('nop')
            return {'jobs': {'tbd': define_asset_job(name='tbd', selection='foo')}}

def test_source_assets():
    if False:
        while True:
            i = 10
    foo = SourceAsset(key=AssetKey('foo'))
    bar = SourceAsset(key=AssetKey('bar'))

    @repository
    def my_repo():
        if False:
            while True:
                i = 10
        return [foo, bar]
    assert my_repo.source_assets_by_key == {AssetKey('foo'): SourceAsset(key=AssetKey('foo')), AssetKey('bar'): SourceAsset(key=AssetKey('bar'))}

def test_direct_assets():
    if False:
        return 10

    @io_manager(required_resource_keys={'foo'})
    def the_manager():
        if False:
            while True:
                i = 10
        pass
    foo_resource = ResourceDefinition.hardcoded_resource('foo')
    foo = SourceAsset('foo', io_manager_def=the_manager, resource_defs={'foo': foo_resource})

    @asset(resource_defs={'foo': foo_resource})
    def asset1():
        if False:
            print('Hello World!')
        pass

    @asset
    def asset2():
        if False:
            return 10
        pass

    @repository
    def my_repo():
        if False:
            i = 10
            return i + 15
        return [foo, asset1, asset2]
    assert len(my_repo.get_all_jobs()) == 1
    assert set(my_repo.get_all_jobs()[0].asset_layer.asset_keys) == {AssetKey(['asset1']), AssetKey(['asset2'])}
    assert my_repo.get_all_jobs()[0].resource_defs['foo'] == foo_resource

def test_direct_assets_duplicate_keys():
    if False:
        print('Hello World!')

    def make_asset():
        if False:
            print('Hello World!')

        @asset
        def asset1():
            if False:
                i = 10
                return i + 15
            pass
        return asset1
    with pytest.raises(DagsterInvalidDefinitionError, match="Duplicate asset key: AssetKey\\(\\['asset1'\\]\\)"):

        @repository
        def my_repo():
            if False:
                return 10
            return [make_asset(), make_asset()]

def test_direct_asset_unsatified_resource():
    if False:
        for i in range(10):
            print('nop')

    @asset(required_resource_keys={'a'})
    def asset1():
        if False:
            while True:
                i = 10
        pass
    with pytest.raises(DagsterInvalidDefinitionError, match="resource with key 'a' required by op 'asset1' was not provided."):

        @repository
        def my_repo():
            if False:
                while True:
                    i = 10
            return [asset1]

def test_direct_asset_unsatified_resource_transitive():
    if False:
        for i in range(10):
            print('nop')

    @resource(required_resource_keys={'b'})
    def resource1():
        if False:
            return 10
        pass

    @asset(resource_defs={'a': resource1})
    def asset1():
        if False:
            print('Hello World!')
        pass
    with pytest.raises(DagsterInvalidDefinitionError, match="resource with key 'b' required by resource with key 'a' was not provided."):

        @repository
        def my_repo():
            if False:
                while True:
                    i = 10
            return [asset1]

def test_source_asset_unsatisfied_resource():
    if False:
        for i in range(10):
            print('nop')

    @io_manager(required_resource_keys={'foo'})
    def the_manager():
        if False:
            return 10
        pass
    with pytest.raises(DagsterInvalidDefinitionError, match="resource with key 'foo' required by resource with key 'foo__io_manager' was not provided."):

        @repository
        def the_repo():
            if False:
                for i in range(10):
                    print('nop')
            return [SourceAsset('foo', io_manager_def=the_manager)]

def test_source_asset_unsatisfied_resource_transitive():
    if False:
        return 10

    @io_manager(required_resource_keys={'foo'})
    def the_manager():
        if False:
            while True:
                i = 10
        pass

    @resource(required_resource_keys={'bar'})
    def foo_resource():
        if False:
            while True:
                i = 10
        pass
    with pytest.raises(DagsterInvalidDefinitionError, match="resource with key 'bar' required by resource with key 'foo' was not provided."):

        @repository
        def the_repo():
            if False:
                print('Hello World!')
            return [SourceAsset('foo', io_manager_def=the_manager, resource_defs={'foo': foo_resource})]

def test_direct_asset_resource_conflicts():
    if False:
        for i in range(10):
            print('nop')

    @asset(resource_defs={'foo': ResourceDefinition.hardcoded_resource('1')})
    def first():
        if False:
            while True:
                i = 10
        pass

    @asset(resource_defs={'foo': ResourceDefinition.hardcoded_resource('2')})
    def second():
        if False:
            print('Hello World!')
        pass
    with pytest.raises(DagsterInvalidDefinitionError, match="Conflicting versions of resource with key 'foo' were provided to different assets."):

        @repository
        def the_repo():
            if False:
                print('Hello World!')
            return [first, second]

def test_source_asset_resource_conflicts():
    if False:
        for i in range(10):
            print('nop')

    @asset(resource_defs={'foo': ResourceDefinition.hardcoded_resource('1')})
    def the_asset():
        if False:
            return 10
        pass

    @io_manager(required_resource_keys={'foo'})
    def the_manager():
        if False:
            while True:
                i = 10
        pass
    the_source = SourceAsset(key=AssetKey('the_key'), io_manager_def=the_manager, resource_defs={'foo': ResourceDefinition.hardcoded_resource('2')})
    with pytest.raises(DagsterInvalidDefinitionError, match="Conflicting versions of resource with key 'foo' were provided to different assets."):

        @repository
        def the_repo():
            if False:
                while True:
                    i = 10
            return [the_asset, the_source]
    other_source = SourceAsset(key=AssetKey('other_key'), io_manager_def=the_manager, resource_defs={'foo': ResourceDefinition.hardcoded_resource('3')})
    with pytest.raises(DagsterInvalidDefinitionError, match="Conflicting versions of resource with key 'foo' were provided to different assets."):

        @repository
        def other_repo():
            if False:
                i = 10
                return i + 15
            return [other_source, the_source]

def test_assets_different_io_manager_defs():
    if False:
        i = 10
        return i + 15

    class MyIOManager(IOManager):

        def handle_output(self, context, obj):
            if False:
                for i in range(10):
                    print('nop')
            assert obj == 10

        def load_input(self, context):
            if False:
                for i in range(10):
                    print('nop')
            return 5
    the_manager_used = []

    @io_manager
    def the_manager():
        if False:
            i = 10
            return i + 15
        the_manager_used.append('yes')
        return MyIOManager()
    other_manager_used = []

    @io_manager
    def other_manager():
        if False:
            for i in range(10):
                print('nop')
        other_manager_used.append('yes')
        return MyIOManager()

    @asset(io_manager_def=the_manager)
    def the_asset(the_source, other_source):
        if False:
            return 10
        return the_source + other_source

    @asset(io_manager_def=other_manager)
    def other_asset(the_source, other_source):
        if False:
            for i in range(10):
                print('nop')
        return the_source + other_source
    the_source = SourceAsset(key=AssetKey('the_source'), io_manager_def=the_manager)
    other_source = SourceAsset(key=AssetKey('other_source'), io_manager_def=other_manager)

    @repository
    def the_repo():
        if False:
            print('Hello World!')
        return [the_asset, other_asset, the_source, other_source]
    assert len(the_repo.get_all_jobs()) == 1
    assert the_repo.get_all_jobs()[0].execute_in_process().success
    assert len(the_manager_used) == 2
    assert len(other_manager_used) == 2

def _create_graph_with_name(name):
    if False:
        print('Hello World!')

    @graph(name=name)
    def _the_graph():
        if False:
            for i in range(10):
                print('nop')
        pass
    return _the_graph

def _create_job_with_name(name):
    if False:
        for i in range(10):
            print('nop')

    @job(name=name)
    def _the_job():
        if False:
            while True:
                i = 10
        pass
    return _the_job

def _create_schedule_from_target(target):
    if False:
        return 10

    @schedule(job=target, cron_schedule='* * * * *')
    def _the_schedule():
        if False:
            print('Hello World!')
        pass
    return _the_schedule

def _create_sensor_from_target(target):
    if False:
        while True:
            i = 10

    @sensor(job=target)
    def _the_sensor():
        if False:
            while True:
                i = 10
        pass
    return _the_sensor

def test_duplicate_graph_valid():
    if False:
        while True:
            i = 10
    the_graph = _create_graph_with_name('foo')

    @repository
    def the_repo_dupe_graph_valid():
        if False:
            print('Hello World!')
        return [the_graph, _create_sensor_from_target(the_graph)]
    assert len(the_repo_dupe_graph_valid.get_all_jobs()) == 1

def test_duplicate_graph_target_invalid():
    if False:
        for i in range(10):
            print('nop')
    the_graph = _create_graph_with_name('foo')
    other_graph = _create_graph_with_name('foo')
    with pytest.raises(DagsterInvalidDefinitionError, match="sensor '_the_sensor' targets graph 'foo', but a different graph with the same name was provided."):

        @repository
        def the_repo_dupe_graph_invalid_sensor():
            if False:
                return 10
            return [the_graph, _create_sensor_from_target(other_graph)]
    with pytest.raises(DagsterInvalidDefinitionError, match="schedule '_the_schedule' targets graph 'foo', but a different graph with the same name was provided."):

        @repository
        def the_repo_dupe_graph_invalid_schedule():
            if False:
                for i in range(10):
                    print('nop')
            return [the_graph, _create_schedule_from_target(other_graph)]

def test_duplicate_unresolved_job_valid():
    if False:
        print('Hello World!')
    the_job = define_asset_job(name='foo')

    @asset
    def foo_asset():
        if False:
            i = 10
            return i + 15
        return 1

    @repository
    def the_repo_dupe_unresolved_job_valid():
        if False:
            return 10
        return [the_job, _create_sensor_from_target(the_job), foo_asset]
    assert len(the_repo_dupe_unresolved_job_valid.get_all_jobs()) == 2

def test_duplicate_unresolved_job_target_invalid():
    if False:
        print('Hello World!')
    the_job = define_asset_job(name='foo')
    other_job = define_asset_job(name='foo', selection='foo')

    @asset
    def foo():
        if False:
            i = 10
            return i + 15
        return None
    with pytest.raises(DagsterInvalidDefinitionError, match="sensor '_the_sensor' targets unresolved asset job 'foo', but a different unresolved asset job with the same name was provided."):

        @repository
        def the_repo_dupe_graph_invalid_sensor():
            if False:
                print('Hello World!')
            return [foo, the_job, _create_sensor_from_target(other_job)]
    with pytest.raises(DagsterInvalidDefinitionError, match="schedule '_the_schedule' targets unresolved asset job 'foo', but a different unresolved asset job with the same name was provided."):

        @repository
        def the_repo_dupe_graph_invalid_schedule():
            if False:
                for i in range(10):
                    print('nop')
            return [foo, the_job, _create_schedule_from_target(other_job)]

def test_duplicate_job_target_valid():
    if False:
        print('Hello World!')
    the_job = _create_job_with_name('foo')

    @repository
    def the_repo_dupe_job_valid():
        if False:
            while True:
                i = 10
        return [the_job, _create_schedule_from_target(the_job), _create_sensor_from_target(the_job)]

def test_duplicate_job_target_invalid():
    if False:
        for i in range(10):
            print('nop')
    the_job = _create_job_with_name('foo')
    other_job = _create_job_with_name('foo')
    with pytest.raises(DagsterInvalidDefinitionError, match="sensor '_the_sensor' targets job 'foo', but a different job with the same name was provided."):

        @repository
        def the_repo_dupe_job_invalid_sensor():
            if False:
                return 10
            return [the_job, _create_sensor_from_target(other_job)]
    with pytest.raises(DagsterInvalidDefinitionError, match="schedule '_the_schedule' targets job 'foo', but a different job with the same name was provided."):

        @repository
        def the_repo_dupe_job_invalid_schedule():
            if False:
                return 10
            return [the_job, _create_schedule_from_target(other_job)]

def test_dupe_jobs_valid():
    if False:
        while True:
            i = 10
    the_job = _create_job_with_name('foo')

    @repository
    def the_repo_dupe_jobs_valid():
        if False:
            for i in range(10):
                print('nop')
        return [the_job, _create_schedule_from_target(the_job), _create_sensor_from_target(the_job)]

def test_dupe_jobs_invalid():
    if False:
        for i in range(10):
            print('nop')
    the_job = _create_job_with_name('foo')
    other_job = _create_job_with_name('foo')
    with pytest.raises(DagsterInvalidDefinitionError, match="schedule '_the_schedule' targets job 'foo', but a different job with the same name was provided."):

        @repository
        def the_repo_dupe_jobs_invalid_schedule():
            if False:
                return 10
            return [the_job, _create_schedule_from_target(other_job)]
    with pytest.raises(DagsterInvalidDefinitionError, match="sensor '_the_sensor' targets job 'foo', but a different job with the same name was provided."):

        @repository
        def the_repo_dupe_jobs_invalid_sensor():
            if False:
                i = 10
                return i + 15
            return [the_job, _create_sensor_from_target(other_job)]

def test_default_executor_repo():
    if False:
        for i in range(10):
            print('nop')

    @repository(default_executor_def=in_process_executor)
    def the_repo():
        if False:
            for i in range(10):
                print('nop')
        return []

def test_default_executor_assets_repo():
    if False:
        return 10

    @graph
    def no_executor_provided():
        if False:
            for i in range(10):
                print('nop')
        pass

    @asset
    def the_asset():
        if False:
            i = 10
            return i + 15
        pass

    @repository(default_executor_def=in_process_executor)
    def the_repo():
        if False:
            print('Hello World!')
        return [no_executor_provided, the_asset]
    assert the_repo.get_job('__ASSET_JOB').executor_def == in_process_executor
    assert the_repo.get_job('no_executor_provided').executor_def == in_process_executor

def test_default_executor_jobs():
    if False:
        i = 10
        return i + 15

    @asset
    def the_asset():
        if False:
            i = 10
            return i + 15
        pass
    unresolved_job = define_asset_job('asset_job', selection='*')

    @executor
    def custom_executor(_):
        if False:
            i = 10
            return i + 15
        pass

    @executor
    def other_custom_executor(_):
        if False:
            print('Hello World!')
        pass

    @job(executor_def=custom_executor)
    def op_job_with_executor():
        if False:
            while True:
                i = 10
        pass

    @job
    def op_job_no_executor():
        if False:
            for i in range(10):
                print('nop')
        pass

    @job(executor_def=multi_or_in_process_executor)
    def job_explicitly_specifies_default_executor():
        if False:
            return 10
        pass

    @job
    def the_job():
        if False:
            return 10
        pass

    @repository(default_executor_def=other_custom_executor)
    def the_repo():
        if False:
            for i in range(10):
                print('nop')
        return [the_asset, op_job_with_executor, op_job_no_executor, unresolved_job, job_explicitly_specifies_default_executor]
    assert the_repo.get_job('asset_job').executor_def == other_custom_executor
    assert the_repo.get_job('op_job_with_executor').executor_def == custom_executor
    assert the_repo.get_job('op_job_no_executor').executor_def == other_custom_executor
    assert the_repo.get_job('job_explicitly_specifies_default_executor').executor_def == multi_or_in_process_executor

def test_list_load():
    if False:
        print('Hello World!')

    @asset
    def asset1():
        if False:
            for i in range(10):
                print('nop')
        return 1

    @asset
    def asset2():
        if False:
            i = 10
            return i + 15
        return 2
    source = SourceAsset(key=AssetKey('a_source_asset'))
    all_assets: Sequence[AssetsDefinition, SourceAsset] = [asset1, asset2, source]

    @repository
    def assets_repo():
        if False:
            i = 10
            return i + 15
        return [all_assets]
    assert len(assets_repo.get_all_jobs()) == 1
    assert set(assets_repo.get_all_jobs()[0].asset_layer.asset_keys) == {AssetKey(['asset1']), AssetKey(['asset2'])}

    @op
    def op1():
        if False:
            while True:
                i = 10
        return 1

    @op
    def op2():
        if False:
            return 10
        return 1

    @job
    def job1():
        if False:
            print('Hello World!')
        op1()

    @job
    def job2():
        if False:
            print('Hello World!')
        op2()
    job_list = [job1, job2]

    @repository
    def job_repo():
        if False:
            for i in range(10):
                print('nop')
        return [job_list]
    assert len(job_repo.get_all_jobs()) == len(job_list)

    @asset
    def asset3():
        if False:
            while True:
                i = 10
        return 3

    @op
    def op3():
        if False:
            i = 10
            return i + 15
        return 3

    @job
    def job3():
        if False:
            for i in range(10):
                print('nop')
        op3()
    combo_list = [asset3, job3]

    @repository
    def combo_repo():
        if False:
            return 10
        return [combo_list]
    assert len(combo_repo.get_all_jobs()) == 2
    assert set(combo_repo.get_all_jobs()[0].asset_layer.asset_keys) == {AssetKey(['asset3'])}

def test_multi_nested_list():
    if False:
        print('Hello World!')

    @asset
    def asset1():
        if False:
            i = 10
            return i + 15
        return 1

    @asset
    def asset2():
        if False:
            for i in range(10):
                print('nop')
        return 2
    source = SourceAsset(key=AssetKey('a_source_asset'))
    layer_1: Sequence[AssetsDefinition, SourceAsset] = [asset2, source]
    layer_2 = [layer_1, asset1]
    with pytest.raises(DagsterInvalidDefinitionError, match='Bad return value from repository'):

        @repository
        def assets_repo():
            if False:
                while True:
                    i = 10
            return [layer_2]

def test_default_executor_config():
    if False:
        while True:
            i = 10

    @asset
    def some_asset():
        if False:
            return 10
        pass

    @repository(default_executor_def=in_process_executor)
    def the_repo():
        if False:
            print('Hello World!')
        return [define_asset_job('the_job', config={'execution': {'config': {'retries': {'enabled': {}}}}}), some_asset]
    assert the_repo.get_job('the_job').executor_def == in_process_executor

def test_scheduled_partitioned_asset_job():
    if False:
        i = 10
        return i + 15
    partitions_def = DailyPartitionsDefinition(start_date='2022-06-06')

    @asset(partitions_def=partitions_def)
    def asset1():
        if False:
            while True:
                i = 10
        ...

    @repository
    def repo():
        if False:
            print('Hello World!')
        return [asset1, build_schedule_from_partitioned_job(define_asset_job('fdsjk', partitions_def=partitions_def))]
    repo.load_all_definitions()

def test_default_loggers_repo():
    if False:
        return 10

    @logger
    def basic():
        if False:
            while True:
                i = 10
        pass

    @repository(default_logger_defs={'foo': basic})
    def the_repo():
        if False:
            for i in range(10):
                print('nop')
        return []

def test_default_loggers_assets_repo():
    if False:
        print('Hello World!')

    @graph
    def no_logger_provided():
        if False:
            print('Hello World!')
        pass

    @asset
    def the_asset():
        if False:
            return 10
        pass

    @logger
    def basic():
        if False:
            for i in range(10):
                print('nop')
        pass

    @repository(default_logger_defs={'foo': basic})
    def the_repo():
        if False:
            i = 10
            return i + 15
        return [no_logger_provided, the_asset]
    assert the_repo.get_job('__ASSET_JOB').loggers == {'foo': basic}
    assert the_repo.get_job('no_logger_provided').loggers == {'foo': basic}

def test_default_loggers_for_jobs():
    if False:
        return 10

    @asset
    def the_asset():
        if False:
            i = 10
            return i + 15
        pass
    unresolved_job = define_asset_job('asset_job', selection='*')

    @logger
    def custom_logger(_):
        if False:
            i = 10
            return i + 15
        pass

    @logger
    def other_custom_logger(_):
        if False:
            while True:
                i = 10
        pass

    @job(logger_defs={'bar': custom_logger})
    def job_with_loggers():
        if False:
            while True:
                i = 10
        pass

    @job
    def job_no_loggers():
        if False:
            return 10
        pass

    @job(logger_defs=default_loggers())
    def job_explicitly_specifies_default_loggers():
        if False:
            for i in range(10):
                print('nop')
        pass

    @repository(default_logger_defs={'foo': other_custom_logger})
    def the_repo():
        if False:
            return 10
        return [the_asset, job_with_loggers, job_no_loggers, unresolved_job, job_explicitly_specifies_default_loggers]
    assert the_repo.get_job('asset_job').loggers == {'foo': other_custom_logger}
    assert the_repo.get_job('job_with_loggers').loggers == {'bar': custom_logger}
    assert the_repo.get_job('job_no_loggers').loggers == {'foo': other_custom_logger}
    assert the_repo.get_job('job_explicitly_specifies_default_loggers').loggers == default_loggers()

def test_default_loggers_keys_conflict():
    if False:
        print('Hello World!')

    @logger
    def some_logger():
        if False:
            i = 10
            return i + 15
        pass

    @logger
    def other_logger():
        if False:
            for i in range(10):
                print('nop')
        pass

    @job(logger_defs={'foo': some_logger})
    def the_job():
        if False:
            while True:
                i = 10
        pass

    @repository(default_logger_defs={'foo': other_logger})
    def the_repo():
        if False:
            while True:
                i = 10
        return [the_job]
    assert the_repo.get_job('the_job').loggers == {'foo': some_logger}

def test_base_jobs():
    if False:
        return 10

    @asset
    def asset1():
        if False:
            while True:
                i = 10
        ...

    @asset(partitions_def=StaticPartitionsDefinition(['a', 'b', 'c']))
    def asset2():
        if False:
            for i in range(10):
                print('nop')
        ...

    @asset(partitions_def=StaticPartitionsDefinition(['x', 'y', 'z']))
    def asset3():
        if False:
            return 10
        ...

    @repository
    def repo():
        if False:
            i = 10
            return i + 15
        return [asset1, asset2, asset3]
    assert sorted(repo.get_implicit_asset_job_names()) == ['__ASSET_JOB_0', '__ASSET_JOB_1']
    assert repo.get_implicit_job_def_for_assets([asset1.key, asset2.key]).asset_layer.asset_keys == {asset1.key, asset2.key}
    assert repo.get_implicit_job_def_for_assets([asset2.key, asset3.key]) is None