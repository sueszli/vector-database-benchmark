import pickle
from typing import Any, Callable, Tuple
import pytest
from dagster import ConfigurableResource, GraphIn, GraphOut, IAttachDifferentObjectToOpContext, In, Int, Out, Output, StaticPartitionsDefinition, VersionStrategy, asset, graph, job, materialize, op, resource, with_resources
from dagster._core.definitions.assets import AssetsDefinition
from dagster._core.definitions.source_asset import SourceAsset
from dagster._core.test_utils import instance_for_test
from dagster._legacy import build_assets_job
from dagster_aws.s3.io_manager import S3PickleIOManager, s3_pickle_io_manager
from dagster_aws.s3.utils import construct_s3_client

class S3TestResource(ConfigurableResource, IAttachDifferentObjectToOpContext):

    def get_client(self) -> Any:
        if False:
            return 10
        return construct_s3_client(max_attempts=5)

    def get_object_to_set_on_execution_context(self) -> Any:
        if False:
            for i in range(10):
                print('nop')
        return self.get_client()

@resource
def s3_test_resource(_):
    if False:
        for i in range(10):
            print('nop')
    return construct_s3_client(max_attempts=5)

@pytest.fixture(name='s3_and_io_manager', params=[True, False])
def s3_and_io_manager_fixture(request) -> Tuple[Any, Callable[[Any], Any]]:
    if False:
        i = 10
        return i + 15
    if request.param:
        return (s3_test_resource, lambda _: s3_pickle_io_manager)
    else:
        return (S3TestResource(), lambda s3: S3PickleIOManager.configure_at_launch(s3_resource=s3))

def define_inty_job(s3_resource, s3_io_manager_builder):
    if False:
        print('Hello World!')

    @op(out=Out(Int))
    def return_one():
        if False:
            return 10
        return 1

    @op(ins={'num': In(Int)}, out=Out(Int))
    def add_one(num):
        if False:
            return 10
        return num + 1

    @job(resource_defs={'io_manager': s3_io_manager_builder(s3_resource), 's3': s3_resource})
    def basic_external_plan_execution():
        if False:
            return 10
        add_one(return_one())
    return basic_external_plan_execution

def test_s3_pickle_io_manager_execution(mock_s3_bucket, s3_and_io_manager):
    if False:
        print('Hello World!')
    assert not len(list(mock_s3_bucket.objects.all()))
    (s3_resource, s3_io_manager_builder) = s3_and_io_manager
    inty_job = define_inty_job(s3_resource, s3_io_manager_builder)
    run_config = {'resources': {'io_manager': {'config': {'s3_bucket': mock_s3_bucket.name}}}}
    result = inty_job.execute_in_process(run_config)
    assert result.output_for_node('return_one') == 1
    assert result.output_for_node('add_one') == 2
    assert len(list(mock_s3_bucket.objects.all())) == 2

def define_multiple_output_job():
    if False:
        return 10

    @op(out={'foo': Out(Int), 'foobar': Out(Int)})
    def return_two_outputs():
        if False:
            print('Hello World!')
        yield Output(10, 'foobar')
        yield Output(5, 'foo')

    @job(resource_defs={'io_manager': s3_pickle_io_manager, 's3': s3_test_resource})
    def output_prefix_execution_plan():
        if False:
            for i in range(10):
                print('nop')
        return_two_outputs()
    return output_prefix_execution_plan

def test_s3_pickle_io_manager_prefix(mock_s3_bucket):
    if False:
        print('Hello World!')
    assert not len(list(mock_s3_bucket.objects.all()))
    prefixy_job = define_multiple_output_job()
    run_config = {'resources': {'io_manager': {'config': {'s3_bucket': mock_s3_bucket.name}}}}
    result = prefixy_job.execute_in_process(run_config)
    assert result.output_for_node('return_two_outputs', 'foo') == 5
    assert result.output_for_node('return_two_outputs', 'foobar') == 10
    assert len(list(mock_s3_bucket.objects.all())) == 2

def test_memoization_s3_io_manager(mock_s3_bucket):
    if False:
        i = 10
        return i + 15

    class BasicVersionStrategy(VersionStrategy):

        def get_op_version(self, _):
            if False:
                i = 10
                return i + 15
            return 'foo'

    @op
    def basic():
        if False:
            print('Hello World!')
        return 'foo'

    @job(resource_defs={'io_manager': s3_pickle_io_manager, 's3': s3_test_resource}, version_strategy=BasicVersionStrategy())
    def memoized():
        if False:
            while True:
                i = 10
        basic()
    run_config = {'resources': {'io_manager': {'config': {'s3_bucket': mock_s3_bucket.name}}}}
    with instance_for_test() as instance:
        result = memoized.execute_in_process(run_config=run_config, instance=instance)
        assert result.success
        assert result.output_for_node('basic') == 'foo'
        result = memoized.execute_in_process(run_config=run_config, instance=instance)
        assert result.success
        assert len(result.all_node_events) == 0

def define_assets_job(bucket):
    if False:
        print('Hello World!')

    @op
    def first_op(first_input):
        if False:
            for i in range(10):
                print('nop')
        assert first_input == 4
        return first_input * 2

    @op
    def second_op(second_input):
        if False:
            return 10
        assert second_input == 8
        return second_input + 3
    source1 = SourceAsset('source1', partitions_def=StaticPartitionsDefinition(['foo', 'bar']))

    @asset
    def asset1(source1):
        if False:
            while True:
                i = 10
        return source1['foo'] + source1['bar']

    @asset
    def asset2(asset1):
        if False:
            print('Hello World!')
        assert asset1 == 3
        return asset1 + 1

    @graph(ins={'asset2': GraphIn()}, out={'asset3': GraphOut()})
    def graph_asset(asset2):
        if False:
            return 10
        return second_op(first_op(asset2))

    @asset(partitions_def=StaticPartitionsDefinition(['apple', 'orange']))
    def partitioned():
        if False:
            return 10
        return 8
    return build_assets_job(name='assets', assets=[asset1, asset2, AssetsDefinition.from_graph(graph_asset), partitioned], source_assets=[source1], resource_defs={'io_manager': s3_pickle_io_manager.configured({'s3_bucket': bucket}), 's3': s3_test_resource})

def test_s3_pickle_io_manager_asset_execution(mock_s3_bucket):
    if False:
        return 10
    assert not len(list(mock_s3_bucket.objects.all()))
    inty_job = define_assets_job(mock_s3_bucket.name)
    mock_s3_bucket.put_object(Key='dagster/source1/foo', Body=pickle.dumps(1))
    mock_s3_bucket.put_object(Key='dagster/source1/bar', Body=pickle.dumps(2))
    result = inty_job.execute_in_process(partition_key='apple')
    assert result.output_for_node('asset1') == 3
    assert result.output_for_node('asset2') == 4
    assert result.output_for_node('graph_asset.first_op') == 8
    assert result.output_for_node('graph_asset.second_op') == 11
    objects = list(mock_s3_bucket.objects.all())
    assert len(objects) == 7
    assert {(o.bucket_name, o.key) for o in objects} == {('test-bucket', 'dagster/source1/bar'), ('test-bucket', 'dagster/source1/foo'), ('test-bucket', 'dagster/asset1'), ('test-bucket', 'dagster/asset2'), ('test-bucket', 'dagster/asset3'), ('test-bucket', 'dagster/partitioned/apple'), ('test-bucket', '/'.join(['dagster', 'storage', result.run_id, 'graph_asset.first_op', 'result']))}

def test_nothing(mock_s3_bucket):
    if False:
        for i in range(10):
            print('nop')

    @asset
    def asset1() -> None:
        if False:
            while True:
                i = 10
        ...

    @asset(deps=[asset1])
    def asset2() -> None:
        if False:
            while True:
                i = 10
        ...
    result = materialize(with_resources([asset1, asset2], resource_defs={'io_manager': s3_pickle_io_manager.configured({'s3_bucket': mock_s3_bucket.name}), 's3': s3_test_resource}))
    handled_output_events = list(filter(lambda evt: evt.is_handled_output, result.all_node_events))
    assert len(handled_output_events) == 2
    for event in handled_output_events:
        assert len(event.event_specific_data.metadata) == 0