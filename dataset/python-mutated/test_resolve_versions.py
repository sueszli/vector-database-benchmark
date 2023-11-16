import hashlib
import pytest
from dagster import Bool, DagsterInvariantViolationError, Float, In, Int, IOManagerDefinition, Out, Output, SourceHashVersionStrategy, String, dagster_type_loader, execute_job, fs_io_manager, graph, io_manager, job, op, reconstructable, resource, usable_as_dagster_type
from dagster._core.definitions.version_strategy import VersionStrategy
from dagster._core.execution.api import create_execution_plan
from dagster._core.execution.plan.outputs import StepOutputHandle
from dagster._core.execution.resolve_versions import join_and_hash, resolve_config_version
from dagster._core.storage.input_manager import input_manager
from dagster._core.storage.memoizable_io_manager import MemoizableIOManager
from dagster._core.storage.tags import MEMOIZED_RUN_TAG
from dagster._core.system_config.objects import ResolvedRunConfig
from dagster._core.test_utils import instance_for_test

class VersionedInMemoryIOManager(MemoizableIOManager):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.values = {}

    def _get_keys(self, context):
        if False:
            return 10
        return (context.step_key, context.name, context.version)

    def handle_output(self, context, obj):
        if False:
            for i in range(10):
                print('nop')
        keys = self._get_keys(context)
        self.values[keys] = obj

    def load_input(self, context):
        if False:
            i = 10
            return i + 15
        keys = self._get_keys(context.upstream_output)
        return self.values[keys]

    def has_output(self, context):
        if False:
            while True:
                i = 10
        keys = self._get_keys(context)
        return keys in self.values

def test_join_and_hash():
    if False:
        for i in range(10):
            print('nop')
    assert join_and_hash('foo') == hashlib.sha1(b'foo').hexdigest()
    assert join_and_hash('foo', None, 'bar') is None
    assert join_and_hash('foo', 'bar') == hashlib.sha1(b'barfoo').hexdigest()
    assert join_and_hash('foo', 'bar', 'zab') == join_and_hash('zab', 'bar', 'foo')

def test_resolve_config_version():
    if False:
        for i in range(10):
            print('nop')
    assert resolve_config_version(None) == join_and_hash()
    assert resolve_config_version({}) == join_and_hash()
    assert resolve_config_version({'a': 'b', 'c': 'd'}) == join_and_hash('a' + join_and_hash('b'), 'c' + join_and_hash('d'))
    assert resolve_config_version({'a': 'b', 'c': 'd'}) == resolve_config_version({'c': 'd', 'a': 'b'})
    assert resolve_config_version({'a': {'b': 'c'}, 'd': 'e'}) == join_and_hash('a' + join_and_hash('b' + join_and_hash('c')), 'd' + join_and_hash('e'))

@op(version='42')
def versioned_op_no_input(_):
    if False:
        for i in range(10):
            print('nop')
    return 4

@op(version='5')
def versioned_op_takes_input(_, intput):
    if False:
        print('Hello World!')
    return 2 * intput

def versioned_job_factory(manager=VersionedInMemoryIOManager()):
    if False:
        return 10

    @job(resource_defs={'io_manager': IOManagerDefinition.hardcoded_io_manager(manager)}, tags={MEMOIZED_RUN_TAG: 'true'})
    def versioned_job():
        if False:
            print('Hello World!')
        versioned_op_takes_input(versioned_op_no_input())
    return versioned_job

@op
def op_takes_input(_, intput):
    if False:
        i = 10
        return i + 15
    return 2 * intput

def partially_versioned_job_factory(manager=VersionedInMemoryIOManager()):
    if False:
        return 10

    @job(resource_defs={'io_manager': IOManagerDefinition.hardcoded_io_manager(manager)}, tags={MEMOIZED_RUN_TAG: 'true'})
    def partially_versioned_job():
        if False:
            while True:
                i = 10
        op_takes_input(versioned_op_no_input())
    return partially_versioned_job

@op
def basic_op(_):
    if False:
        for i in range(10):
            print('nop')
    return 5

@op
def basic_takes_input_op(_, intpt):
    if False:
        while True:
            i = 10
    return intpt * 4

@job
def no_version_job():
    if False:
        print('Hello World!')
    basic_takes_input_op(basic_op())

def test_memoized_plan_no_memoized_results():
    if False:
        return 10
    with instance_for_test() as instance:
        versioned_job = versioned_job_factory()
        memoized_plan = create_execution_plan(versioned_job, instance_ref=instance.get_ref())
        assert set(memoized_plan.step_keys_to_execute) == {'versioned_op_no_input', 'versioned_op_takes_input'}

def test_memoized_plan_memoized_results():
    if False:
        print('Hello World!')
    with instance_for_test() as instance:
        manager = VersionedInMemoryIOManager()
        versioned_job = versioned_job_factory(manager)
        plan = create_execution_plan(versioned_job, instance_ref=instance.get_ref())
        resolved_run_config = ResolvedRunConfig.build(versioned_job)
        step_output_handle = StepOutputHandle('versioned_op_no_input', 'result')
        step_output_version = plan.get_version_for_step_output_handle(step_output_handle)
        manager.values[step_output_handle.step_key, step_output_handle.output_name, step_output_version] = 4
        memoized_plan = plan.build_memoized_plan(versioned_job, resolved_run_config, instance=None, selected_step_keys=None)
        assert memoized_plan.step_keys_to_execute == ['versioned_op_takes_input']

def test_memoization_no_code_version_for_op():
    if False:
        while True:
            i = 10
    with instance_for_test() as instance:
        partially_versioned_job = partially_versioned_job_factory()
        with pytest.raises(DagsterInvariantViolationError, match="While using memoization, version for op 'op_takes_input' was None. Please either provide a versioning strategy for your job, or provide a version using the op decorator."):
            create_execution_plan(partially_versioned_job, instance_ref=instance.get_ref())

def _get_ext_version(config_value):
    if False:
        print('Hello World!')
    return join_and_hash(str(config_value))

@dagster_type_loader(String, loader_version='97', external_version_fn=_get_ext_version)
def InputHydration(_, _hello):
    if False:
        for i in range(10):
            print('nop')
    return 'Hello'

@usable_as_dagster_type(loader=InputHydration)
class CustomType(str):
    pass

def test_externally_loaded_inputs():
    if False:
        while True:
            i = 10
    for (type_to_test, type_value) in [(String, ('foo', 'bar')), (Int, (int(42), int(46))), (Float, (float(5.42), float(5.45))), (Bool, (False, True)), (CustomType, ('bar', 'baz'))]:
        run_test_with_builtin_type(type_to_test, type_value)

def run_test_with_builtin_type(type_to_test, type_values):
    if False:
        for i in range(10):
            print('nop')
    (first_type_val, second_type_val) = type_values
    manager = VersionedInMemoryIOManager()

    @op(version='42', ins={'_builtin_type': In(type_to_test)})
    def op_ext_input(_builtin_type):
        if False:
            while True:
                i = 10
        pass

    @job(resource_defs={'io_manager': IOManagerDefinition.hardcoded_io_manager(manager)}, tags={MEMOIZED_RUN_TAG: 'true'})
    def my_job():
        if False:
            i = 10
            return i + 15
        versioned_op_takes_input(op_ext_input())
    input_config = {'_builtin_type': first_type_val}
    run_config = {'ops': {'op_ext_input': {'inputs': input_config}}}
    with instance_for_test() as instance:
        unmemoized_plan = create_execution_plan(my_job, run_config=run_config, instance_ref=instance.get_ref())
        assert len(unmemoized_plan.step_keys_to_execute) == 2
        step_output_handle = StepOutputHandle('op_ext_input', 'result')
        version = unmemoized_plan.get_version_for_step_output_handle(step_output_handle)
        manager.values[step_output_handle.step_key, step_output_handle.output_name, version] = 5
        memoized_plan = create_execution_plan(my_job, run_config=run_config, instance_ref=instance.get_ref())
        assert memoized_plan.step_keys_to_execute == ['versioned_op_takes_input']
        input_config['_builtin_type'] = second_type_val
        unmemoized_plan = create_execution_plan(my_job, run_config=run_config, instance_ref=instance.get_ref())
        assert len(unmemoized_plan.step_keys_to_execute) == 2

def test_memoized_plan_default_input_val():
    if False:
        for i in range(10):
            print('nop')

    @op(version='42', ins={'_my_input': In(String, default_value='DEFAULTVAL')})
    def op_default_input(_my_input):
        if False:
            for i in range(10):
                print('nop')
        pass

    @job(resource_defs={'io_manager': IOManagerDefinition.hardcoded_io_manager(VersionedInMemoryIOManager())}, tags={MEMOIZED_RUN_TAG: 'true'})
    def job_default_value():
        if False:
            for i in range(10):
                print('nop')
        op_default_input()
    with instance_for_test() as instance:
        unmemoized_plan = create_execution_plan(job_default_value, instance_ref=instance.get_ref())
        assert unmemoized_plan.step_keys_to_execute == ['op_default_input']

def test_memoized_plan_affected_by_resource_config():
    if False:
        return 10

    @op(required_resource_keys={'my_resource'}, version='39')
    def op_reqs_resource():
        if False:
            return 10
        pass

    @resource(version='42', config_schema={'foo': str})
    def basic():
        if False:
            i = 10
            return i + 15
        pass
    manager = VersionedInMemoryIOManager()

    @job(resource_defs={'my_resource': basic, 'io_manager': IOManagerDefinition.hardcoded_io_manager(manager)}, tags={MEMOIZED_RUN_TAG: 'true'})
    def my_job():
        if False:
            i = 10
            return i + 15
        op_reqs_resource()
    with instance_for_test() as instance:
        my_resource_config = {'foo': 'bar'}
        run_config = {'resources': {'my_resource': {'config': my_resource_config}}}
        unmemoized_plan = create_execution_plan(my_job, run_config=run_config, instance_ref=instance.get_ref())
        assert unmemoized_plan.step_keys_to_execute == ['op_reqs_resource']
        step_output_handle = StepOutputHandle('op_reqs_resource', 'result')
        version = unmemoized_plan.get_version_for_step_output_handle(step_output_handle)
        manager.values[step_output_handle.step_key, step_output_handle.output_name, version] = 5
        memoized_plan = create_execution_plan(my_job, run_config=run_config, instance_ref=instance.get_ref())
        assert len(memoized_plan.step_keys_to_execute) == 0
        my_resource_config['foo'] = 'baz'
        changed_config_plan = create_execution_plan(my_job, run_config=run_config, instance_ref=instance.get_ref())
        assert changed_config_plan.step_keys_to_execute == ['op_reqs_resource']

def test_memoized_plan_custom_io_manager_key():
    if False:
        while True:
            i = 10
    manager = VersionedInMemoryIOManager()
    mgr_def = IOManagerDefinition.hardcoded_io_manager(manager)

    @op(version='39', out=Out(io_manager_key='my_key'))
    def op_requires_io_manager():
        if False:
            for i in range(10):
                print('nop')
        return Output(5)

    @job(resource_defs={'my_key': mgr_def}, tags={MEMOIZED_RUN_TAG: 'true'})
    def io_mgr_job():
        if False:
            while True:
                i = 10
        op_requires_io_manager()
    with instance_for_test() as instance:
        unmemoized_plan = create_execution_plan(io_mgr_job, instance_ref=instance.get_ref())
        assert unmemoized_plan.step_keys_to_execute == ['op_requires_io_manager']
        step_output_handle = StepOutputHandle('op_requires_io_manager', 'result')
        version = unmemoized_plan.get_version_for_step_output_handle(step_output_handle)
        manager.values[step_output_handle.step_key, step_output_handle.output_name, version] = 5
        memoized_plan = create_execution_plan(io_mgr_job, instance_ref=instance.get_ref())
        assert len(memoized_plan.step_keys_to_execute) == 0

def test_unmemoized_inner_op():
    if False:
        while True:
            i = 10

    @op
    def op_no_version():
        if False:
            while True:
                i = 10
        pass

    @graph
    def wrap():
        if False:
            for i in range(10):
                print('nop')
        return op_no_version()

    @job(resource_defs={'fake': IOManagerDefinition.hardcoded_io_manager(VersionedInMemoryIOManager())}, tags={MEMOIZED_RUN_TAG: 'true'})
    def wrap_job():
        if False:
            for i in range(10):
                print('nop')
        wrap()
    with instance_for_test() as instance:
        with pytest.raises(DagsterInvariantViolationError, match="While using memoization, version for op 'op_no_version' was None. Please either provide a versioning strategy for your job, or provide a version using the op decorator."):
            create_execution_plan(wrap_job, instance_ref=instance.get_ref())

def test_memoized_inner_op():
    if False:
        print('Hello World!')

    @op(version='versioned')
    def op_versioned():
        if False:
            while True:
                i = 10
        pass

    @graph
    def wrap():
        if False:
            for i in range(10):
                print('nop')
        return op_versioned()
    mgr = VersionedInMemoryIOManager()

    @job(resource_defs={'io_manager': IOManagerDefinition.hardcoded_io_manager(mgr)}, tags={MEMOIZED_RUN_TAG: 'true'})
    def wrap_job():
        if False:
            for i in range(10):
                print('nop')
        wrap()
    with instance_for_test() as instance:
        unmemoized_plan = create_execution_plan(wrap_job, instance_ref=instance.get_ref())
        step_output_handle = StepOutputHandle('wrap.op_versioned', 'result')
        assert unmemoized_plan.step_keys_to_execute == [step_output_handle.step_key]
        step_output_version = unmemoized_plan.get_version_for_step_output_handle(step_output_handle)
        mgr.values[step_output_handle.step_key, step_output_handle.output_name, step_output_version] = 4
        memoized_plan = unmemoized_plan.build_memoized_plan(wrap_job, ResolvedRunConfig.build(wrap_job), instance=None, selected_step_keys=None)
        assert len(memoized_plan.step_keys_to_execute) == 0

def test_configured_versions():
    if False:
        print('Hello World!')

    @op(version='5')
    def op_to_configure():
        if False:
            for i in range(10):
                print('nop')
        pass
    assert op_to_configure.configured({}, name='op_has_been_configured').version == '5'

    @resource(version='5')
    def resource_to_configure(_):
        if False:
            while True:
                i = 10
        pass
    assert resource_to_configure.configured({}).version == '5'

def test_memoized_plan_inits_resources_once():
    if False:
        return 10

    @op(out=Out(io_manager_key='foo'), version='foo')
    def foo_op():
        if False:
            i = 10
            return i + 15
        pass

    @op(out=Out(io_manager_key='bar'), version='bar')
    def bar_op():
        if False:
            print('Hello World!')
        pass
    foo_capture = []
    bar_capture = []
    resource_dep_capture = []
    default_capture = []

    @io_manager(required_resource_keys={'my_resource'})
    def foo_manager():
        if False:
            for i in range(10):
                print('nop')
        foo_capture.append('entered')
        return VersionedInMemoryIOManager()

    @io_manager(required_resource_keys={'my_resource'})
    def bar_manager():
        if False:
            for i in range(10):
                print('nop')
        bar_capture.append('entered')
        return VersionedInMemoryIOManager()

    @io_manager
    def default_manager():
        if False:
            print('Hello World!')
        default_capture.append('entered')
        return VersionedInMemoryIOManager()

    @resource
    def my_resource():
        if False:
            return 10
        resource_dep_capture.append('entered')
        return None

    @job(resource_defs={'foo': foo_manager, 'bar': bar_manager, 'my_resource': my_resource, 'io_manager': default_manager}, tags={MEMOIZED_RUN_TAG: 'true'})
    def wrap_job():
        if False:
            i = 10
            return i + 15
        foo_op()
        foo_op.alias('another_foo')()
        bar_op()
        bar_op.alias('another_bar')()
    with instance_for_test() as instance:
        create_execution_plan(wrap_job, instance_ref=instance.get_ref())
    assert len(foo_capture) == 1
    assert len(bar_capture) == 1
    assert len(resource_dep_capture) == 1
    assert len(default_capture) == 0

def test_memoized_plan_disable_memoization():
    if False:
        print('Hello World!')

    @op(version='hello')
    def my_op():
        if False:
            while True:
                i = 10
        return 5
    mgr = VersionedInMemoryIOManager()

    @job(resource_defs={'io_manager': IOManagerDefinition.hardcoded_io_manager(mgr)}, tags={MEMOIZED_RUN_TAG: 'true'})
    def my_job():
        if False:
            print('Hello World!')
        my_op()
    with instance_for_test() as instance:
        unmemoized_plan = create_execution_plan(my_job, instance_ref=instance.get_ref())
        assert len(unmemoized_plan.step_keys_to_execute) == 1
        step_output_handle = StepOutputHandle('my_op', 'result')
        version = unmemoized_plan.get_version_for_step_output_handle(step_output_handle)
        mgr.values[step_output_handle.step_key, step_output_handle.output_name, version] = 5
        memoized_plan = create_execution_plan(my_job, instance_ref=instance.get_ref())
        assert len(memoized_plan.step_keys_to_execute) == 0
        unmemoized_again = create_execution_plan(my_job, instance_ref=instance.get_ref(), tags={MEMOIZED_RUN_TAG: 'false'})
        assert len(unmemoized_again.step_keys_to_execute) == 1

def test_memoized_plan_input_manager():
    if False:
        return 10

    @input_manager(version='foo')
    def my_input_manager():
        if False:
            while True:
                i = 10
        return 5

    @op(ins={'x': In(input_manager_key='my_input_manager')}, version='foo')
    def my_op_takes_input(x):
        if False:
            i = 10
            return i + 15
        return x

    @job(resource_defs={'io_manager': IOManagerDefinition.hardcoded_io_manager(VersionedInMemoryIOManager()), 'my_input_manager': my_input_manager}, tags={MEMOIZED_RUN_TAG: 'true'})
    def my_job():
        if False:
            for i in range(10):
                print('nop')
        my_op_takes_input()
    with instance_for_test() as instance:
        plan = create_execution_plan(my_job, instance_ref=instance.get_ref())
        assert plan.get_version_for_step_output_handle(StepOutputHandle('my_op_takes_input', 'result')) is not None

def test_memoized_plan_input_manager_input_config():
    if False:
        return 10

    @input_manager(version='foo', input_config_schema={'my_str': str})
    def my_input_manager():
        if False:
            print('Hello World!')
        return 5

    @op(ins={'x': In(input_manager_key='my_input_manager')}, version='foo')
    def my_op_takes_input(x):
        if False:
            i = 10
            return i + 15
        return x

    @job(resource_defs={'io_manager': IOManagerDefinition.hardcoded_io_manager(VersionedInMemoryIOManager()), 'my_input_manager': my_input_manager}, tags={MEMOIZED_RUN_TAG: 'true'})
    def my_job():
        if False:
            while True:
                i = 10
        my_op_takes_input()
    input_config = {'my_str': 'foo'}
    run_config = {'ops': {'my_op_takes_input': {'inputs': {'x': input_config}}}}
    with instance_for_test() as instance:
        plan = create_execution_plan(my_job, instance_ref=instance.get_ref(), run_config=run_config)
        output_version = plan.get_version_for_step_output_handle(StepOutputHandle('my_op_takes_input', 'result'))
        assert output_version is not None
        input_config['my_str'] = 'bar'
        plan = create_execution_plan(my_job, instance_ref=instance.get_ref(), run_config=run_config)
        new_output_version = plan.get_version_for_step_output_handle(StepOutputHandle('my_op_takes_input', 'result'))
        assert not new_output_version == output_version

def test_memoized_plan_input_manager_resource_config():
    if False:
        while True:
            i = 10

    @input_manager(version='foo', config_schema={'my_str': str})
    def my_input_manager():
        if False:
            for i in range(10):
                print('nop')
        return 5

    @op(ins={'x': In(input_manager_key='my_input_manager')}, version='foo')
    def my_op_takes_input(x):
        if False:
            i = 10
            return i + 15
        return x

    @job(resource_defs={'io_manager': IOManagerDefinition.hardcoded_io_manager(VersionedInMemoryIOManager()), 'my_input_manager': my_input_manager}, tags={MEMOIZED_RUN_TAG: 'true'})
    def my_job():
        if False:
            for i in range(10):
                print('nop')
        my_op_takes_input()
    resource_config = {'my_str': 'foo'}
    run_config = {'resources': {'my_input_manager': {'config': resource_config}}}
    with instance_for_test() as instance:
        plan = create_execution_plan(my_job, instance_ref=instance.get_ref(), run_config=run_config)
        output_version = plan.get_version_for_step_output_handle(StepOutputHandle('my_op_takes_input', 'result'))
        assert output_version is not None
        resource_config['my_str'] = 'bar'
        plan = create_execution_plan(my_job, instance_ref=instance.get_ref(), run_config=run_config)
        new_output_version = plan.get_version_for_step_output_handle(StepOutputHandle('my_op_takes_input', 'result'))
        assert not new_output_version == output_version
bad_str = "'well this doesn't work !'"

class BadopStrategy(VersionStrategy):

    def get_op_version(self, _):
        if False:
            print('Hello World!')
        return bad_str

    def get_resource_version(self, _):
        if False:
            for i in range(10):
                print('nop')
        return 'foo'

class BadResourceStrategy(VersionStrategy):

    def get_op_version(self, _):
        if False:
            for i in range(10):
                print('nop')
        return 'foo'

    def get_resource_version(self, _):
        if False:
            while True:
                i = 10
        return bad_str

def get_basic_graph():
    if False:
        print('Hello World!')

    @op
    def my_op():
        if False:
            print('Hello World!')
        pass

    @graph
    def my_graph():
        if False:
            i = 10
            return i + 15
        my_op()
    return my_graph

def get_graph_reqs_resource():
    if False:
        print('Hello World!')

    @op(required_resource_keys={'foo'})
    def my_op():
        if False:
            i = 10
            return i + 15
        pass

    @graph
    def my_graph():
        if False:
            for i in range(10):
                print('nop')
        my_op()
    return my_graph

def get_graph_reqs_input_manager():
    if False:
        for i in range(10):
            print('nop')

    @op(ins={'x': In(input_manager_key='my_key')})
    def my_op(x):
        if False:
            return 10
        return x

    @graph
    def my_graph():
        if False:
            while True:
                i = 10
        my_op()
    return my_graph

@pytest.mark.parametrize('graph_for_test,strategy', [(get_basic_graph(), BadopStrategy()), (get_graph_reqs_resource(), BadResourceStrategy()), (get_graph_reqs_input_manager(), BadResourceStrategy())])
def test_bad_version_str(graph_for_test, strategy):
    if False:
        i = 10
        return i + 15

    @resource
    def my_resource():
        if False:
            return 10
        pass

    @input_manager
    def my_manager():
        if False:
            i = 10
            return i + 15
        pass
    with instance_for_test() as instance:
        my_job = graph_for_test.to_job(version_strategy=strategy, resource_defs={'io_manager': IOManagerDefinition.hardcoded_io_manager(VersionedInMemoryIOManager()), 'my_key': my_manager, 'foo': my_resource})
        with pytest.raises(DagsterInvariantViolationError, match=f"'{bad_str}' is not a valid version string."):
            create_execution_plan(my_job, instance_ref=instance.get_ref())

def get_version_strategy_job():
    if False:
        return 10

    @op
    def my_op():
        if False:
            i = 10
            return i + 15
        return 5

    class MyVersionStrategy(VersionStrategy):

        def get_op_version(self, _):
            if False:
                for i in range(10):
                    print('nop')
            return 'foo'

    @job(version_strategy=MyVersionStrategy(), resource_defs={'io_manager': fs_io_manager})
    def ten_job():
        if False:
            print('Hello World!')
        my_op()
    return ten_job

def test_version_strategy_on_job():
    if False:
        while True:
            i = 10
    ten_job = get_version_strategy_job()
    with instance_for_test() as instance:
        result = ten_job.execute_in_process(instance=instance)
        assert result.success
        memoized_plan = create_execution_plan(ten_job, instance_ref=instance.get_ref())
        assert len(memoized_plan.step_keys_to_execute) == 0

def test_version_strategy_no_resource_version():
    if False:
        for i in range(10):
            print('nop')

    @op(required_resource_keys={'foo'})
    def my_op(context):
        if False:
            print('Hello World!')
        return context.resources.foo

    @resource
    def foo_resource():
        if False:
            for i in range(10):
                print('nop')
        return 'bar'

    class MyVersionStrategy(VersionStrategy):

        def get_op_version(self, _):
            if False:
                for i in range(10):
                    print('nop')
            return 'foo'

    @job(version_strategy=MyVersionStrategy(), resource_defs={'io_manager': IOManagerDefinition.hardcoded_io_manager(VersionedInMemoryIOManager()), 'foo': foo_resource})
    def my_job():
        if False:
            for i in range(10):
                print('nop')
        my_op()
    with instance_for_test() as instance:
        my_job.execute_in_process(instance=instance)
        memoized_plan = create_execution_plan(my_job, instance_ref=instance.get_ref())
        assert len(memoized_plan.step_keys_to_execute) == 0

def test_code_versioning_strategy():
    if False:
        i = 10
        return i + 15

    @op
    def my_op():
        if False:
            return 10
        return 5

    @job(version_strategy=SourceHashVersionStrategy())
    def call_the_op():
        if False:
            while True:
                i = 10
        my_op()
    with instance_for_test() as instance:
        result = call_the_op.execute_in_process(instance=instance)
        assert result.success
        memoized_plan = create_execution_plan(call_the_op, instance_ref=instance.get_ref())
        assert len(memoized_plan.step_keys_to_execute) == 0

def test_memoization_multiprocess_execution():
    if False:
        while True:
            i = 10
    with instance_for_test() as instance:
        result = execute_job(reconstructable(get_version_strategy_job), instance)
        assert result.success
        memoized_plan = create_execution_plan(get_version_strategy_job(), instance_ref=instance.get_ref())
        assert len(memoized_plan.step_keys_to_execute) == 0

def test_source_hash_with_input_manager():
    if False:
        while True:
            i = 10

    @input_manager
    def my_input_manager():
        if False:
            while True:
                i = 10
        return 5

    @op(ins={'x': In(input_manager_key='manager')})
    def the_op(x):
        if False:
            while True:
                i = 10
        return x + 1

    @job(version_strategy=SourceHashVersionStrategy(), resource_defs={'manager': my_input_manager})
    def call_the_op():
        if False:
            return 10
        the_op()
    with instance_for_test() as instance:
        result = call_the_op.execute_in_process(instance=instance)
        assert result.success
        memoized_plan = create_execution_plan(call_the_op, instance_ref=instance.get_ref())
        assert len(memoized_plan.step_keys_to_execute) == 0