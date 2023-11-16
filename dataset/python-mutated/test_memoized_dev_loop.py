import tempfile
from typing import Any, Mapping, Sequence
import pytest
from dagster import DagsterInvariantViolationError, DynamicOut, DynamicOutput, In, graph, op, resource
from dagster._core.definitions.job_definition import JobDefinition
from dagster._core.definitions.reconstruct import reconstructable
from dagster._core.definitions.version_strategy import VersionStrategy
from dagster._core.execution.api import ReexecutionOptions, create_execution_plan, execute_job
from dagster._core.instance import DagsterInstance
from dagster._core.storage.input_manager import input_manager
from dagster._core.storage.memoizable_io_manager import versioned_filesystem_io_manager
from dagster._core.storage.tags import MEMOIZED_RUN_TAG
from dagster._core.test_utils import instance_for_test
from .memoized_dev_loop_job import op_job

def get_step_keys_to_execute(job_def: JobDefinition, run_config: Mapping[str, Any], instance: DagsterInstance) -> Sequence[str]:
    if False:
        print('Hello World!')
    return create_execution_plan(job_def, run_config, instance_ref=instance.get_ref(), tags={MEMOIZED_RUN_TAG: 'true'}).step_keys_to_execute

def test_dev_loop_changing_versions():
    if False:
        return 10
    with tempfile.TemporaryDirectory() as temp_dir:
        with instance_for_test(temp_dir=temp_dir) as instance:
            run_config = {'ops': {'create_string_1_asset_op': {'config': {'input_str': 'apple'}}, 'take_string_1_asset_op': {'config': {'input_str': 'apple'}}}, 'resources': {'io_manager': {'config': {'base_dir': temp_dir}}}}
            with execute_job(reconstructable(op_job), run_config=run_config, tags={MEMOIZED_RUN_TAG: 'true'}, instance=instance) as result:
                assert result.success
            assert not get_step_keys_to_execute(op_job, run_config, instance)
            run_config['ops']['take_string_1_asset_op']['config']['input_str'] = 'banana'
            assert get_step_keys_to_execute(op_job, run_config, instance) == ['take_string_1_asset_op']
            with execute_job(reconstructable(op_job), reexecution_options=ReexecutionOptions(parent_run_id=result.run_id), run_config=run_config, tags={MEMOIZED_RUN_TAG: 'true'}, instance=instance) as result:
                assert result.success
            assert not get_step_keys_to_execute(op_job, run_config, instance)
            with execute_job(reconstructable(op_job), run_config=run_config, tags={MEMOIZED_RUN_TAG: 'true'}, instance=instance) as result:
                assert result.success
                assert len(result.all_node_events) == 0

def test_memoization_with_default_strategy():
    if False:
        i = 10
        return i + 15
    recorder = []

    @resource()
    def my_resource():
        if False:
            return 10
        pass

    @op(required_resource_keys={'my_resource'})
    def my_op():
        if False:
            print('Hello World!')
        recorder.append('entered')

    @graph
    def my_graph():
        if False:
            print('Hello World!')
        my_op()

    class MyVersionStrategy(VersionStrategy):

        def get_op_version(self, _):
            if False:
                return 10
            return 'foo'

        def get_resource_version(self, _):
            if False:
                for i in range(10):
                    print('nop')
            return 'foo'
    with tempfile.TemporaryDirectory() as temp_dir:
        with instance_for_test(temp_dir=temp_dir) as instance:
            my_job = my_graph.to_job(version_strategy=MyVersionStrategy(), resource_defs={'io_manager': versioned_filesystem_io_manager.configured({'base_dir': temp_dir}), 'my_resource': my_resource})
            unmemoized_plan = create_execution_plan(my_job, instance_ref=instance.get_ref())
            assert len(unmemoized_plan.step_keys_to_execute) == 1
            result = my_job.execute_in_process(instance=instance)
            assert result.success
            assert len(recorder) == 1
            execution_plan = create_execution_plan(my_job, instance_ref=instance.get_ref())
            assert len(execution_plan.step_keys_to_execute) == 0
            result = my_job.execute_in_process(instance=instance)
            assert result.success
            assert len(recorder) == 1

def test_memoization_with_step_selection():
    if False:
        i = 10
        return i + 15

    @op
    def op1():
        if False:
            print('Hello World!')
        pass

    @op
    def op2(arg1):
        if False:
            while True:
                i = 10
        del arg1

    @graph
    def my_graph():
        if False:
            print('Hello World!')
        op2(op1())

    class MyVersionStrategy(VersionStrategy):

        def get_op_version(self, context):
            if False:
                print('Hello World!')
            if context.op_def.name == op1.name:
                return 'foo'
            else:
                import uuid
                return str(uuid.uuid4()).replace('-', '_')
    with tempfile.TemporaryDirectory() as temp_dir:
        with instance_for_test(temp_dir=temp_dir) as instance:
            my_job = my_graph.to_job(version_strategy=MyVersionStrategy(), resource_defs={'io_manager': versioned_filesystem_io_manager.configured({'base_dir': temp_dir})}, tags={MEMOIZED_RUN_TAG: 'True'})
            single_op_selected_plan = create_execution_plan(my_job, instance_ref=instance.get_ref(), step_keys_to_execute=['op1'])
            assert len(single_op_selected_plan.step_keys_to_execute) == 1
            assert single_op_selected_plan.step_keys_to_execute == ['op1']
            result = my_job.execute_in_process(instance=instance)
            assert result.success
            assert create_execution_plan(my_job, instance_ref=instance.get_ref(), step_keys_to_execute=['op1']).step_keys_to_execute == []
            assert create_execution_plan(my_job, instance_ref=instance.get_ref(), step_keys_to_execute=['op2']).step_keys_to_execute == ['op2']

def test_memoization_with_default_strategy_overriden():
    if False:
        while True:
            i = 10
    version = ['foo']

    class MyVersionStrategy(VersionStrategy):

        def get_op_version(self, _):
            if False:
                i = 10
                return i + 15
            return version[0]
    recorder = []

    @op(version='override')
    def my_op():
        if False:
            for i in range(10):
                print('nop')
        recorder.append('entered')

    @graph
    def my_graph():
        if False:
            for i in range(10):
                print('nop')
        my_op()
    with tempfile.TemporaryDirectory() as temp_dir:
        with instance_for_test(temp_dir=temp_dir) as instance:
            my_job = my_graph.to_job(version_strategy=MyVersionStrategy(), resource_defs={'io_manager': versioned_filesystem_io_manager.configured({'base_dir': temp_dir})})
            unmemoized_plan = create_execution_plan(my_job, instance_ref=instance.get_ref())
            assert len(unmemoized_plan.step_keys_to_execute) == 1
            result = my_job.execute_in_process(instance=instance)
            assert result.success
            assert len(recorder) == 1
            version.remove('foo')
            version.append('bar')
            memoized_plan = create_execution_plan(my_job, instance_ref=instance.get_ref())
            assert len(memoized_plan.step_keys_to_execute) == 0
            result = my_job.execute_in_process(instance=instance)
            assert result.success
            assert len(recorder) == 1
            unmemoized_plan = create_execution_plan(my_job, instance_ref=instance.get_ref(), tags={MEMOIZED_RUN_TAG: 'false'})
            assert len(unmemoized_plan.step_keys_to_execute) == 1

def test_version_strategy_depends_from_context():
    if False:
        for i in range(10):
            print('nop')
    version = {'foo': 'bar'}
    version_strategy_called = []
    graph_executed = []

    class ContextDependantVersionStrategy(VersionStrategy):

        def get_op_version(self, context):
            if False:
                i = 10
                return i + 15
            version_strategy_called.append('versioned')
            solid_arg = context.op_config['arg']
            return version[solid_arg]

        def get_resource_version(self, context):
            if False:
                for i in range(10):
                    print('nop')
            resource_arg = context.resource_config['arg']
            return version[resource_arg]
    run_config = {'ops': {'my_op': {'config': {'arg': 'foo'}}}}

    @op
    def my_op():
        if False:
            return 10
        graph_executed.append('executed')

    @graph
    def my_graph():
        if False:
            i = 10
            return i + 15
        my_op()
    with tempfile.TemporaryDirectory() as temp_dir:
        with instance_for_test(temp_dir=temp_dir) as instance:
            my_job = my_graph.to_job(version_strategy=ContextDependantVersionStrategy(), resource_defs={'io_manager': versioned_filesystem_io_manager.configured({'base_dir': temp_dir})})
            result = my_job.execute_in_process(run_config=run_config, instance=instance)
            assert result.success
            assert len(graph_executed) > 0
            assert len(version_strategy_called) > 0
            graph_executed = []
            version_strategy_called = []
            result = my_job.execute_in_process(run_config=run_config, instance=instance)
            assert result.success
            assert len(graph_executed) == 0
            assert len(version_strategy_called) > 0
            graph_executed = []
            version_strategy_called = []
            version['foo'] = 'not_bar'
            result = my_job.execute_in_process(run_config=run_config, instance=instance)
            assert result.success
            assert len(graph_executed) > 0
            assert len(version_strategy_called) > 0
            unmemoized_plan = create_execution_plan(my_job, instance_ref=instance.get_ref(), tags={MEMOIZED_RUN_TAG: 'false'})
            assert len(unmemoized_plan.step_keys_to_execute) == 1

def test_version_strategy_input_manager():
    if False:
        return 10

    class MyVersionStrategy(VersionStrategy):

        def get_op_version(self, _):
            if False:
                return 10
            return 'foo'

        def get_resource_version(self, _):
            if False:
                print('Hello World!')
            return 'foo'

    @input_manager
    def my_input_manager(_):
        if False:
            for i in range(10):
                print('nop')
        return 5

    @op(ins={'x': In(input_manager_key='my_input_manager')})
    def my_op(x):
        if False:
            for i in range(10):
                print('nop')
        return x

    @graph
    def my_graph():
        if False:
            i = 10
            return i + 15
        my_op()
    with tempfile.TemporaryDirectory() as temp_dir:
        with instance_for_test(temp_dir=temp_dir) as instance:
            my_job = my_graph.to_job(resource_defs={'io_manager': versioned_filesystem_io_manager, 'my_input_manager': my_input_manager}, version_strategy=MyVersionStrategy())
            result = my_job.execute_in_process(instance=instance)
            assert result.success
            post_memoization_plan = create_execution_plan(my_job, instance_ref=instance.get_ref())
            assert len(post_memoization_plan.step_keys_to_execute) == 0

def test_dynamic_memoization_error():
    if False:
        print('Hello World!')

    class MyVersionStrategy(VersionStrategy):

        def get_op_version(self, _):
            if False:
                for i in range(10):
                    print('nop')
            return 'foo'

        def get_resource_version(self, _):
            if False:
                while True:
                    i = 10
            return 'foo'

    @op(out=DynamicOut())
    def emit():
        if False:
            i = 10
            return i + 15
        yield DynamicOutput(1, mapping_key='one')
        yield DynamicOutput(2, mapping_key='two')

    @op
    def return_input(x):
        if False:
            print('Hello World!')
        return x

    @graph
    def dynamic_graph():
        if False:
            for i in range(10):
                print('nop')
        x = emit().map(return_input)
        return_input(x.collect())

    @graph
    def just_mapping_graph():
        if False:
            i = 10
            return i + 15
        emit().map(return_input)
    with instance_for_test() as instance:
        for cur_graph in [dynamic_graph, just_mapping_graph]:
            with pytest.raises(DagsterInvariantViolationError, match='Attempted to use memoization with dynamic orchestration, which is not yet supported.'):
                my_job = cur_graph.to_job(version_strategy=MyVersionStrategy(), resource_defs={'io_manager': versioned_filesystem_io_manager})
                my_job.execute_in_process(instance=instance)