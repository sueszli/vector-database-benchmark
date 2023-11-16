from contextlib import contextmanager
from enum import Enum as PythonEnum
from unittest import mock
import pytest
from dagster import DagsterEventType, DagsterInvariantViolationError, DagsterResourceFunctionError, Enum, EnumValue, Field, GraphDefinition, Int, ResourceDefinition, String, build_op_context, configured, execute_job, fs_io_manager, graph, job, op, reconstructable, resource
from dagster._core.definitions.job_base import InMemoryJob
from dagster._core.definitions.resource_definition import dagster_maintained_resource, make_values_resource
from dagster._core.errors import DagsterConfigMappingFunctionError, DagsterInvalidDefinitionError
from dagster._core.events.log import EventLogEntry, construct_event_logger
from dagster._core.execution.api import create_execution_plan, execute_plan
from dagster._core.instance import DagsterInstance
from dagster._core.test_utils import instance_for_test
from dagster._core.utils import coerce_valid_log_level

def define_string_resource():
    if False:
        print('Hello World!')
    return ResourceDefinition(config_schema=String, resource_fn=lambda init_context: init_context.resource_config)

def test_resource_decorator_no_context():
    if False:
        i = 10
        return i + 15

    @resource
    def _basic():
        if False:
            for i in range(10):
                print('nop')
        pass

    @resource(required_resource_keys={'foo', 'bar'}, config_schema={'foo': str})
    def _reqs_resources():
        if False:
            for i in range(10):
                print('nop')
        pass

def assert_job_runs_with_resource(resource_def, resource_config, expected_resource):
    if False:
        while True:
            i = 10
    called = {}

    @op(required_resource_keys={'some_name'})
    def a_op(context):
        if False:
            i = 10
            return i + 15
        called['yup'] = True
        assert context.resources.some_name == expected_resource
    job_def = GraphDefinition(name='with_a_resource', node_defs=[a_op]).to_job(resource_defs={'some_name': resource_def})
    run_config = {'resources': {'some_name': {'config': resource_config}}} if resource_config else {}
    result = job_def.execute_in_process(run_config)
    assert result.success
    assert called['yup']

def test_basic_resource():
    if False:
        while True:
            i = 10
    called = {}

    @op(required_resource_keys={'a_string'})
    def a_op(context):
        if False:
            i = 10
            return i + 15
        called['yup'] = True
        assert context.resources.a_string == 'foo'
    job_def = GraphDefinition(name='with_a_resource', node_defs=[a_op]).to_job(resource_defs={'a_string': define_string_resource()})
    result = job_def.execute_in_process({'resources': {'a_string': {'config': 'foo'}}})
    assert result.success
    assert called['yup']

def test_resource_with_dependencies():
    if False:
        for i in range(10):
            print('nop')
    called = {}

    @resource
    def foo_resource(_):
        if False:
            i = 10
            return i + 15
        called['foo_resource'] = True
        return 'foo'

    @resource(required_resource_keys={'foo_resource'})
    def bar_resource(init_context):
        if False:
            while True:
                i = 10
        called['bar_resource'] = True
        return init_context.resources.foo_resource + 'bar'

    @op(required_resource_keys={'bar_resource'})
    def dep_op(context):
        if False:
            print('Hello World!')
        called['dep_op'] = True
        assert context.resources.bar_resource == 'foobar'
    job_def = GraphDefinition(name='with_dep_resource', node_defs=[dep_op]).to_job(resource_defs={'foo_resource': foo_resource, 'bar_resource': bar_resource})
    result = job_def.execute_in_process()
    assert result.success
    assert called['foo_resource']
    assert called['bar_resource']
    assert called['dep_op']

def test_resource_cyclic_dependencies():
    if False:
        print('Hello World!')
    called = {}

    @resource(required_resource_keys={'bar_resource'})
    def foo_resource(init_context):
        if False:
            i = 10
            return i + 15
        called['foo_resource'] = True
        return init_context.resources.bar_resource + 'foo'

    @resource(required_resource_keys={'foo_resource'})
    def bar_resource(init_context):
        if False:
            return 10
        called['bar_resource'] = True
        return init_context.resources.foo_resource + 'bar'

    @op(required_resource_keys={'bar_resource'})
    def dep_op(context):
        if False:
            return 10
        called['dep_op'] = True
        assert context.resources.bar_resource == 'foobar'
    with pytest.raises(DagsterInvariantViolationError, match='Resource key "(foo_resource|bar_resource)" transitively depends on itself.'):
        GraphDefinition(name='with_dep_resource', node_defs=[dep_op]).to_job(resource_defs={'foo_resource': foo_resource, 'bar_resource': bar_resource})

def test_yield_resource():
    if False:
        for i in range(10):
            print('nop')
    called = {}

    @op(required_resource_keys={'a_string'})
    def a_op(context):
        if False:
            for i in range(10):
                print('nop')
        called['yup'] = True
        assert context.resources.a_string == 'foo'

    def _do_resource(init_context):
        if False:
            i = 10
            return i + 15
        yield init_context.resource_config
    yield_string_resource = ResourceDefinition(config_schema=String, resource_fn=_do_resource)
    job_def = GraphDefinition(name='with_a_yield_resource', node_defs=[a_op]).to_job(resource_defs={'a_string': yield_string_resource})
    result = job_def.execute_in_process({'resources': {'a_string': {'config': 'foo'}}})
    assert result.success
    assert called['yup']

def test_yield_multiple_resources():
    if False:
        return 10
    called = {}
    saw = []

    @op(required_resource_keys={'string_one', 'string_two'})
    def a_op(context):
        if False:
            while True:
                i = 10
        called['yup'] = True
        assert context.resources.string_one == 'foo'
        assert context.resources.string_two == 'bar'

    def _do_resource(init_context):
        if False:
            while True:
                i = 10
        saw.append('before yield ' + init_context.resource_config)
        yield init_context.resource_config
        saw.append('after yield ' + init_context.resource_config)
    yield_string_resource = ResourceDefinition(config_schema=String, resource_fn=_do_resource)
    job_def = GraphDefinition(name='with_yield_resources', node_defs=[a_op]).to_job(resource_defs={'string_one': yield_string_resource, 'string_two': yield_string_resource})
    result = job_def.execute_in_process({'resources': {'string_one': {'config': 'foo'}, 'string_two': {'config': 'bar'}}})
    assert result.success
    assert called['yup']
    assert len(saw) == 4
    assert 'before yield' in saw[0]
    assert 'before yield' in saw[1]
    assert 'after yield' in saw[2]
    assert 'after yield' in saw[3]

def test_resource_decorator():
    if False:
        while True:
            i = 10
    called = {}
    saw = []

    @op(required_resource_keys={'string_one', 'string_two'})
    def a_op(context):
        if False:
            i = 10
            return i + 15
        called['yup'] = True
        assert context.resources.string_one == 'foo'
        assert context.resources.string_two == 'bar'

    @resource(config_schema=Field(String))
    def yielding_string_resource(init_context):
        if False:
            print('Hello World!')
        saw.append('before yield ' + init_context.resource_config)
        yield init_context.resource_config
        saw.append('after yield ' + init_context.resource_config)
    job_def = GraphDefinition(name='with_yield_resources', node_defs=[a_op]).to_job(resource_defs={'string_one': yielding_string_resource, 'string_two': yielding_string_resource})
    result = job_def.execute_in_process({'resources': {'string_one': {'config': 'foo'}, 'string_two': {'config': 'bar'}}})
    assert result.success
    assert called['yup']
    assert len(saw) == 4
    assert 'before yield' in saw[0]
    assert 'before yield' in saw[1]
    assert 'after yield' in saw[2]
    assert 'after yield' in saw[3]

def test_mixed_multiple_resources():
    if False:
        for i in range(10):
            print('nop')
    called = {}
    saw = []

    @op(required_resource_keys={'returned_string', 'yielded_string'})
    def a_op(context):
        if False:
            print('Hello World!')
        called['yup'] = True
        assert context.resources.returned_string == 'foo'
        assert context.resources.yielded_string == 'bar'

    def _do_yield_resource(init_context):
        if False:
            print('Hello World!')
        saw.append('before yield ' + init_context.resource_config)
        yield init_context.resource_config
        saw.append('after yield ' + init_context.resource_config)
    yield_string_resource = ResourceDefinition(config_schema=String, resource_fn=_do_yield_resource)

    def _do_return_resource(init_context):
        if False:
            return 10
        saw.append('before return ' + init_context.resource_config)
        return init_context.resource_config
    return_string_resource = ResourceDefinition(config_schema=String, resource_fn=_do_return_resource)
    job_def = GraphDefinition(name='with_a_yield_resource', node_defs=[a_op]).to_job(resource_defs={'yielded_string': yield_string_resource, 'returned_string': return_string_resource})
    result = job_def.execute_in_process({'resources': {'returned_string': {'config': 'foo'}, 'yielded_string': {'config': 'bar'}}})
    assert result.success
    assert called['yup']
    assert 'before yield bar' in saw[0] or 'before return foo' in saw[0]
    assert 'before yield bar' in saw[1] or 'before return foo' in saw[1]
    assert 'after yield bar' in saw[2]

def test_none_resource():
    if False:
        while True:
            i = 10
    called = {}

    @op(required_resource_keys={'test_null'})
    def op_test_null(context):
        if False:
            print('Hello World!')
        assert context.resources.test_null is None
        called['yup'] = True
    job_def = GraphDefinition(name='test_none_resource', node_defs=[op_test_null]).to_job(resource_defs={'test_null': ResourceDefinition.none_resource()})
    result = job_def.execute_in_process()
    assert result.success
    assert called['yup']

def test_string_resource():
    if False:
        for i in range(10):
            print('nop')
    called = {}

    @op(required_resource_keys={'test_string'})
    def op_test_string(context):
        if False:
            return 10
        assert context.resources.test_string == 'foo'
        called['yup'] = True
    job_def = GraphDefinition(name='test_string_resource', node_defs=[op_test_string]).to_job(resource_defs={'test_string': ResourceDefinition.string_resource()})
    result = job_def.execute_in_process({'resources': {'test_string': {'config': 'foo'}}})
    assert result.success
    assert called['yup']

def test_variables_resource():
    if False:
        return 10
    any_variable = 1
    single_variable = {'foo': 'my_string'}
    multi_variables = {'foo': 'my_string', 'bar': 1}

    @op(required_resource_keys={'any_variable', 'single_variable', 'multi_variables'})
    def my_op(context):
        if False:
            while True:
                i = 10
        assert context.resources.any_variable == any_variable
        assert context.resources.single_variable == single_variable
        assert context.resources.multi_variables == multi_variables

    @job(resource_defs={'any_variable': make_values_resource(), 'single_variable': make_values_resource(foo=str), 'multi_variables': make_values_resource(foo=str, bar=int)})
    def my_job():
        if False:
            while True:
                i = 10
        my_op()
    result = my_job.execute_in_process(run_config={'resources': {'any_variable': {'config': any_variable}, 'single_variable': {'config': single_variable}, 'multi_variables': {'config': multi_variables}}})
    assert result.success

def test_hardcoded_resource():
    if False:
        print('Hello World!')
    called = {}
    mock_obj = mock.MagicMock()

    @op(required_resource_keys={'hardcoded'})
    def op_hardcoded(context):
        if False:
            for i in range(10):
                print('nop')
        assert context.resources.hardcoded('called')
        called['yup'] = True
    job_def = GraphDefinition(name='hardcoded_resource', node_defs=[op_hardcoded]).to_job(resource_defs={'hardcoded': ResourceDefinition.hardcoded_resource(mock_obj)})
    result = job_def.execute_in_process()
    assert result.success
    assert called['yup']
    mock_obj.assert_called_with('called')

def test_mock_resource():
    if False:
        return 10
    called = {}

    @op(required_resource_keys={'test_mock'})
    def op_test_mock(context):
        if False:
            for i in range(10):
                print('nop')
        assert context.resources.test_mock is not None
        called['yup'] = True
    job_def = GraphDefinition(name='test_mock_resource', node_defs=[op_test_mock]).to_job(resource_defs={'test_mock': ResourceDefinition.mock_resource()})
    result = job_def.execute_in_process()
    assert result.success
    assert called['yup']

def test_no_config_resource_pass_none():
    if False:
        while True:
            i = 10
    called = {}

    @resource(None)
    def return_thing(_init_context):
        if False:
            for i in range(10):
                print('nop')
        called['resource'] = True
        return 'thing'

    @op(required_resource_keys={'return_thing'})
    def check_thing(context):
        if False:
            while True:
                i = 10
        called['solid'] = True
        assert context.resources.return_thing == 'thing'
    job_def = GraphDefinition(name='test_no_config_resource', node_defs=[check_thing]).to_job(resource_defs={'return_thing': return_thing})
    job_def.execute_in_process()
    assert called['resource']
    assert called['solid']

def test_no_config_resource_no_arg():
    if False:
        for i in range(10):
            print('nop')
    called = {}

    @resource()
    def return_thing(_init_context):
        if False:
            while True:
                i = 10
        called['resource'] = True
        return 'thing'

    @op(required_resource_keys={'return_thing'})
    def check_thing(context):
        if False:
            for i in range(10):
                print('nop')
        called['solid'] = True
        assert context.resources.return_thing == 'thing'
    job_def = GraphDefinition(name='test_no_config_resource', node_defs=[check_thing]).to_job(resource_defs={'return_thing': return_thing})
    job_def.execute_in_process()
    assert called['resource']
    assert called['solid']

def test_no_config_resource_bare_no_arg():
    if False:
        return 10
    called = {}

    @resource
    def return_thing(_init_context):
        if False:
            for i in range(10):
                print('nop')
        called['resource'] = True
        return 'thing'

    @op(required_resource_keys={'return_thing'})
    def check_thing(context):
        if False:
            return 10
        called['solid'] = True
        assert context.resources.return_thing == 'thing'
    job_def = GraphDefinition(name='test_no_config_resource', node_defs=[check_thing]).to_job(resource_defs={'return_thing': return_thing})
    job_def.execute_in_process()
    assert called['resource']
    assert called['solid']

def test_no_config_resource_definition():
    if False:
        for i in range(10):
            print('nop')
    called = {}

    def _return_thing_resource_fn(_init_context):
        if False:
            i = 10
            return i + 15
        called['resource'] = True
        return 'thing'

    @op(required_resource_keys={'return_thing'})
    def check_thing(context):
        if False:
            while True:
                i = 10
        called['solid'] = True
        assert context.resources.return_thing == 'thing'
    job_def = GraphDefinition(name='test_no_config_resource', node_defs=[check_thing]).to_job(resource_defs={'return_thing': ResourceDefinition(_return_thing_resource_fn)})
    job_def.execute_in_process()
    assert called['resource']
    assert called['solid']

def test_resource_cleanup():
    if False:
        return 10
    called = {}

    def _cleanup_resource_fn(_init_context):
        if False:
            for i in range(10):
                print('nop')
        called['creation'] = True
        yield True
        called['cleanup'] = True

    @op(required_resource_keys={'resource_with_cleanup'})
    def check_resource_created(context):
        if False:
            print('Hello World!')
        called['solid'] = True
        assert context.resources.resource_with_cleanup is True
    job_def = GraphDefinition(name='test_resource_cleanup', node_defs=[check_resource_created]).to_job(resource_defs={'resource_with_cleanup': ResourceDefinition(_cleanup_resource_fn)})
    job_def.execute_in_process()
    assert called['creation'] is True
    assert called['solid'] is True
    assert called['cleanup'] is True

def test_stacked_resource_cleanup():
    if False:
        i = 10
        return i + 15
    called = []

    def _cleanup_resource_fn_1(_init_context):
        if False:
            print('Hello World!')
        called.append('creation_1')
        yield True
        called.append('cleanup_1')

    def _cleanup_resource_fn_2(_init_context):
        if False:
            i = 10
            return i + 15
        called.append('creation_2')
        yield True
        called.append('cleanup_2')

    @op(required_resource_keys={'resource_with_cleanup_1', 'resource_with_cleanup_2'})
    def check_resource_created(context):
        if False:
            return 10
        called.append('solid')
        assert context.resources.resource_with_cleanup_1 is True
        assert context.resources.resource_with_cleanup_2 is True
    job_def = GraphDefinition(name='test_resource_cleanup', node_defs=[check_resource_created]).to_job(resource_defs={'resource_with_cleanup_1': ResourceDefinition(_cleanup_resource_fn_1), 'resource_with_cleanup_2': ResourceDefinition(_cleanup_resource_fn_2)})
    job_def.execute_in_process()
    assert called == ['creation_1', 'creation_2', 'solid', 'cleanup_2', 'cleanup_1']

def test_incorrect_resource_init_error():
    if False:
        while True:
            i = 10

    @resource
    def _correct_resource(_):
        if False:
            while True:
                i = 10
        pass

    @resource
    def _correct_resource_no_context():
        if False:
            for i in range(10):
                print('nop')
        pass
    with pytest.raises(DagsterInvalidDefinitionError, match='expects only a single positional required argument. Got required extra params _b, _c'):

        @resource
        def _incorrect_resource_2(_a, _b, _c, _d=4):
            if False:
                return 10
            pass

    @resource
    def _correct_resource_2(_a, _b=1, _c=2):
        if False:
            i = 10
            return i + 15
        pass

def test_resource_init_failure():
    if False:
        i = 10
        return i + 15

    @resource
    def failing_resource(_init_context):
        if False:
            for i in range(10):
                print('nop')
        raise Exception('Uh oh')

    @op(required_resource_keys={'failing_resource'})
    def failing_resource_op(_context):
        if False:
            print('Hello World!')
        pass
    the_job = GraphDefinition(name='test_resource_init_failure', node_defs=[failing_resource_op]).to_job(resource_defs={'failing_resource': failing_resource})
    res = the_job.execute_in_process(raise_on_error=False)
    event_types = [event.event_type_value for event in res.all_events]
    assert DagsterEventType.PIPELINE_FAILURE.value in event_types
    instance = DagsterInstance.ephemeral()
    execution_plan = create_execution_plan(the_job)
    dagster_run = instance.create_run_for_job(the_job, execution_plan=execution_plan)
    with pytest.raises(DagsterResourceFunctionError, match='Error executing resource_fn on ResourceDefinition failing_resource'):
        execute_plan(execution_plan, InMemoryJob(the_job), dagster_run=dagster_run, instance=instance)
    events = the_job.execute_in_process(raise_on_error=False).all_events
    event_types = [event.event_type_value for event in events]
    assert DagsterEventType.PIPELINE_FAILURE.value in event_types

def test_dagster_type_resource_decorator_config():
    if False:
        i = 10
        return i + 15

    @resource(Int)
    def dagster_type_resource_config(_):
        if False:
            print('Hello World!')
        raise Exception('not called')
    assert dagster_type_resource_config.config_schema.config_type.given_name == 'Int'

    @resource(int)
    def python_type_resource_config(_):
        if False:
            print('Hello World!')
        raise Exception('not called')
    assert python_type_resource_config.config_schema.config_type.given_name == 'Int'

def test_resource_init_failure_with_teardown():
    if False:
        return 10
    called = []
    cleaned = []

    @resource
    def resource_a(_):
        if False:
            i = 10
            return i + 15
        try:
            called.append('A')
            yield 'A'
        finally:
            cleaned.append('A')

    @resource
    def resource_b(_):
        if False:
            print('Hello World!')
        try:
            called.append('B')
            raise Exception('uh oh')
            yield 'B'
        finally:
            cleaned.append('B')

    @op(required_resource_keys={'a', 'b'})
    def resource_op(_):
        if False:
            for i in range(10):
                print('nop')
        pass
    job_def = GraphDefinition(name='test_resource_init_failure_with_cleanup', node_defs=[resource_op]).to_job(resource_defs={'a': resource_a, 'b': resource_b})
    res = job_def.execute_in_process(raise_on_error=False)
    event_types = [event.event_type_value for event in res.all_events]
    assert DagsterEventType.PIPELINE_FAILURE.value in event_types
    assert called == ['A', 'B']
    assert cleaned == ['B', 'A']
    called = []
    cleaned = []
    events = job_def.execute_in_process(raise_on_error=False).all_events
    event_types = [event.event_type_value for event in events]
    assert DagsterEventType.PIPELINE_FAILURE.value in event_types
    assert called == ['A', 'B']
    assert cleaned == ['B', 'A']

def test_op_failure_resource_teardown():
    if False:
        for i in range(10):
            print('nop')
    called = []
    cleaned = []

    @resource
    def resource_a(_):
        if False:
            return 10
        try:
            called.append('A')
            yield 'A'
        finally:
            cleaned.append('A')

    @resource
    def resource_b(_):
        if False:
            while True:
                i = 10
        try:
            called.append('B')
            yield 'B'
        finally:
            cleaned.append('B')

    @op(required_resource_keys={'a', 'b'})
    def resource_op(_):
        if False:
            print('Hello World!')
        raise Exception('uh oh')
    job_def = GraphDefinition(name='test_solid_failure_resource_teardown', node_defs=[resource_op]).to_job(resource_defs={'a': resource_a, 'b': resource_b})
    res = job_def.execute_in_process(raise_on_error=False)
    assert res.all_events[-1].event_type_value == 'PIPELINE_FAILURE'
    assert called == ['A', 'B']
    assert cleaned == ['B', 'A']
    called = []
    cleaned = []
    events = job_def.execute_in_process(raise_on_error=False).all_events
    assert len(events) > 1
    assert events[-1].event_type_value == 'PIPELINE_FAILURE'
    assert called == ['A', 'B']
    assert cleaned == ['B', 'A']

def test_op_failure_resource_teardown_raise():
    if False:
        print('Hello World!')
    'Test that teardown is invoked in resources for tests that raise_on_error.'
    called = []
    cleaned = []

    @resource
    def resource_a(_):
        if False:
            for i in range(10):
                print('nop')
        try:
            called.append('A')
            yield 'A'
        finally:
            cleaned.append('A')

    @resource
    def resource_b(_):
        if False:
            print('Hello World!')
        try:
            called.append('B')
            yield 'B'
        finally:
            cleaned.append('B')

    @op(required_resource_keys={'a', 'b'})
    def resource_op(_):
        if False:
            while True:
                i = 10
        raise Exception('uh oh')
    job_def = GraphDefinition(name='test_solid_failure_resource_teardown', node_defs=[resource_op]).to_job(resource_defs={'a': resource_a, 'b': resource_b})
    with pytest.raises(Exception):
        job_def.execute_in_process()
    assert called == ['A', 'B']
    assert cleaned == ['B', 'A']
    called = []
    cleaned = []

def test_resource_teardown_failure():
    if False:
        return 10
    called = []
    cleaned = []

    @resource
    def resource_a(_):
        if False:
            for i in range(10):
                print('nop')
        try:
            called.append('A')
            yield 'A'
        finally:
            cleaned.append('A')

    @resource
    def resource_b(_):
        if False:
            i = 10
            return i + 15
        try:
            called.append('B')
            yield 'B'
        finally:
            raise Exception('uh oh')
            cleaned.append('B')

    @op(required_resource_keys={'a', 'b'})
    def resource_op(_):
        if False:
            i = 10
            return i + 15
        pass
    job_def = GraphDefinition(name='test_resource_teardown_failure', node_defs=[resource_op]).to_job(resource_defs={'a': resource_a, 'b': resource_b})
    result = job_def.execute_in_process(raise_on_error=False)
    assert result.success
    assert len(result.filter_events(lambda evt: evt.is_run_failure)) == 0
    error_events = [event for event in result.all_events if event.event_type == DagsterEventType.ENGINE_EVENT and event.event_specific_data.error]
    assert len(error_events) == 1
    assert called == ['A', 'B']
    assert cleaned == ['A']

def define_resource_teardown_failure_job():
    if False:
        print('Hello World!')

    @resource
    def resource_a(_):
        if False:
            while True:
                i = 10
        try:
            yield 'A'
        finally:
            pass

    @resource
    def resource_b(_):
        if False:
            while True:
                i = 10
        try:
            yield 'B'
        finally:
            raise Exception('uh oh')

    @op(required_resource_keys={'a', 'b'})
    def resource_op(_):
        if False:
            i = 10
            return i + 15
        pass
    return GraphDefinition(name='resource_teardown_failure', node_defs=[resource_op]).to_job(resource_defs={'a': resource_a, 'b': resource_b, 'io_manager': fs_io_manager})

def test_multiprocessing_resource_teardown_failure():
    if False:
        while True:
            i = 10
    with instance_for_test() as instance:
        recon_job = reconstructable(define_resource_teardown_failure_job)
        result = execute_job(recon_job, instance=instance, raise_on_error=False)
        assert result.success
        error_events = [event for event in result.all_events if event.event_type == DagsterEventType.ENGINE_EVENT and event.event_specific_data.error]
        assert len(error_events) == 1

def test_single_step_resource_event_logs():
    if False:
        i = 10
        return i + 15
    USER_SOLID_MESSAGE = 'I AM A SOLID'
    USER_RESOURCE_MESSAGE = 'I AM A RESOURCE'
    events = []

    def event_callback(record):
        if False:
            print('Hello World!')
        assert isinstance(record, EventLogEntry)
        events.append(record)

    @op(required_resource_keys={'a'})
    def resource_op(context):
        if False:
            i = 10
            return i + 15
        context.log.info(USER_SOLID_MESSAGE)

    @resource
    def resource_a(context):
        if False:
            while True:
                i = 10
        context.log.info(USER_RESOURCE_MESSAGE)
        return 'A'
    the_job = GraphDefinition(name='resource_logging_job', node_defs=[resource_op]).to_job(resource_defs={'a': resource_a}, logger_defs={'callback': construct_event_logger(event_callback)})
    result = the_job.execute_in_process(run_config={'loggers': {'callback': {}}}, op_selection=['resource_op'])
    assert result.success
    log_messages = [event for event in events if isinstance(event, EventLogEntry) and event.level == coerce_valid_log_level('INFO')]
    assert len(log_messages) == 2
    resource_log_message = next(iter([message for message in log_messages if message.user_message == USER_RESOURCE_MESSAGE]))
    assert resource_log_message.step_key == 'resource_op'

def test_configured_with_config():
    if False:
        for i in range(10):
            print('nop')
    str_resource = define_string_resource()
    configured_resource = str_resource.configured('foo')
    assert_job_runs_with_resource(configured_resource, {}, 'foo')

def test_configured_with_fn():
    if False:
        for i in range(10):
            print('nop')
    str_resource = define_string_resource()
    configured_resource = str_resource.configured(lambda num: str(num + 1), Int)
    assert_job_runs_with_resource(configured_resource, 2, '3')

def test_configured_decorator_with_fn():
    if False:
        while True:
            i = 10
    str_resource = define_string_resource()

    @configured(str_resource, Int)
    def configured_resource(num):
        if False:
            print('Hello World!')
        return str(num + 1)
    assert_job_runs_with_resource(configured_resource, 2, '3')

def test_configured_decorator_with_fn_and_user_code_error():
    if False:
        while True:
            i = 10
    str_resource = define_string_resource()

    @configured(str_resource, Int)
    def configured_resource(num):
        if False:
            i = 10
            return i + 15
        raise Exception('beep boop broke')
    with pytest.raises(DagsterConfigMappingFunctionError, match='The config mapping function on a `configured` ResourceDefinition has thrown an unexpected error during its execution.') as user_code_exc:
        assert_job_runs_with_resource(configured_resource, 2, 'unreachable')
    assert user_code_exc.value.user_exception.args[0] == 'beep boop broke'

class TestPythonEnum(PythonEnum):
    VALUE_ONE = 0
    OTHER = 1
DagsterEnumType = Enum('ResourceTestEnum', [EnumValue('VALUE_ONE', TestPythonEnum.VALUE_ONE), EnumValue('OTHER', TestPythonEnum.OTHER)])

def test_resource_with_enum_in_schema():
    if False:
        while True:
            i = 10

    @resource(config_schema={'enum': DagsterEnumType})
    def enum_resource(context):
        if False:
            i = 10
            return i + 15
        return context.resource_config['enum']
    assert_job_runs_with_resource(enum_resource, {'enum': 'VALUE_ONE'}, TestPythonEnum.VALUE_ONE)

def test_resource_with_enum_in_schema_configured():
    if False:
        for i in range(10):
            print('nop')

    @resource(config_schema={'enum': DagsterEnumType})
    def enum_resource(context):
        if False:
            for i in range(10):
                print('nop')
        return context.resource_config['enum']

    @configured(enum_resource, {'enum': DagsterEnumType})
    def passthrough_to_enum_resource(config):
        if False:
            for i in range(10):
                print('nop')
        return {'enum': 'VALUE_ONE' if config['enum'] == TestPythonEnum.VALUE_ONE else 'OTHER'}
    assert_job_runs_with_resource(passthrough_to_enum_resource, {'enum': 'VALUE_ONE'}, TestPythonEnum.VALUE_ONE)

def test_resource_run_info_exists_during_execution():
    if False:
        for i in range(10):
            print('nop')

    @resource
    def resource_checks_run_info(init_context):
        if False:
            i = 10
            return i + 15
        assert init_context.dagster_run.run_id == init_context.run_id
        return 1
    assert_job_runs_with_resource(resource_checks_run_info, {}, 1)

def test_resource_needs_resource():
    if False:
        i = 10
        return i + 15

    @resource(required_resource_keys={'bar_resource'})
    def foo_resource(init_context):
        if False:
            i = 10
            return i + 15
        return init_context.resources.bar_resource + 'foo'

    @op(required_resource_keys={'foo_resource'})
    def op_requires_foo():
        if False:
            while True:
                i = 10
        pass
    with pytest.raises(DagsterInvariantViolationError, match="Resource with key 'bar_resource' required by resource with key 'foo_resource', but not provided."):

        @job(resource_defs={'foo_resource': foo_resource})
        def _fail():
            if False:
                print('Hello World!')
            op_requires_foo()

def test_resource_op_subset():
    if False:
        for i in range(10):
            print('nop')

    @resource(required_resource_keys={'bar'})
    def foo_resource(_):
        if False:
            for i in range(10):
                print('nop')
        return 'FOO'

    @resource()
    def bar_resource(_):
        if False:
            print('Hello World!')
        return 'BAR'

    @resource()
    def baz_resource(_):
        if False:
            print('Hello World!')
        return 'BAZ'

    @op(required_resource_keys={'baz'})
    def baz_op(_):
        if False:
            while True:
                i = 10
        pass

    @op(required_resource_keys={'foo'})
    def foo_op(_):
        if False:
            return 10
        pass

    @op(required_resource_keys={'bar'})
    def bar_op(_):
        if False:
            print('Hello World!')
        pass

    @job(resource_defs={'foo': foo_resource, 'baz': baz_resource, 'bar': bar_resource})
    def nested():
        if False:
            while True:
                i = 10
        foo_op()
        bar_op()
        baz_op()
    assert set(nested.get_required_resource_defs().keys()) == {'foo', 'bar', 'baz', 'io_manager'}
    assert nested.get_subset(op_selection=['foo_op']).get_required_resource_defs().keys() == {'foo', 'bar', 'io_manager'}
    assert nested.get_subset(op_selection=['bar_op']).get_required_resource_defs().keys() == {'bar', 'io_manager'}
    assert nested.get_subset(op_selection=['baz_op']).get_required_resource_defs().keys() == {'baz', 'io_manager'}

def test_config_with_no_schema():
    if False:
        i = 10
        return i + 15

    @resource
    def my_resource(init_context):
        if False:
            for i in range(10):
                print('nop')
        return init_context.resource_config

    @op(required_resource_keys={'resource'})
    def my_op(context):
        if False:
            i = 10
            return i + 15
        assert context.resources.resource == 5

    @job(resource_defs={'resource': my_resource})
    def my_job():
        if False:
            return 10
        my_op()
    assert my_job.execute_in_process(run_config={'resources': {'resource': {'config': 5}}}).success

def test_configured_resource_unused():
    if False:
        return 10
    entered = []

    @resource
    def basic_resource(_):
        if False:
            return 10
        pass

    @configured(basic_resource)
    def configured_resource(_):
        if False:
            print('Hello World!')
        entered.append('True')

    @op(required_resource_keys={'bar'})
    def basic_op(_):
        if False:
            return 10
        pass

    @job(resource_defs={'foo': configured_resource, 'bar': basic_resource})
    def basic_job():
        if False:
            i = 10
            return i + 15
        basic_op()
    basic_job.execute_in_process()
    assert not entered

def test_context_manager_resource():
    if False:
        print('Hello World!')
    event_list = []

    @resource
    @contextmanager
    def cm_resource():
        if False:
            print('Hello World!')
        try:
            event_list.append('foo')
            yield 'foo'
        finally:
            event_list.append('finally')

    @op(required_resource_keys={'cm'})
    def basic(context):
        if False:
            i = 10
            return i + 15
        event_list.append('compute')
        assert context.resources.cm == 'foo'
    with build_op_context(resources={'cm': cm_resource}) as context:
        basic(context)
    assert event_list == ['foo', 'compute', 'finally']
    with pytest.raises(DagsterInvariantViolationError, match='At least one provided resource is a generator, but attempting to access resources outside of context manager scope.'):
        basic(build_op_context(resources={'cm': cm_resource}))

    @graph
    def call_basic():
        if False:
            return 10
        basic()
    event_list = []
    assert call_basic.execute_in_process(resources={'cm': cm_resource}).success
    assert event_list == ['foo', 'compute', 'finally']

def test_telemetry_custom_resource():
    if False:
        i = 10
        return i + 15

    class MyResource:

        def foo(self) -> str:
            if False:
                print('Hello World!')
            return 'bar'

    @resource
    def my_resource():
        if False:
            while True:
                i = 10
        return MyResource()
    assert not my_resource._is_dagster_maintained()

def test_telemetry_dagster_io_manager():
    if False:
        for i in range(10):
            print('nop')

    class MyResource:

        def foo(self) -> str:
            if False:
                print('Hello World!')
            return 'bar'

    @dagster_maintained_resource
    @resource
    def my_resource():
        if False:
            for i in range(10):
                print('nop')
        return MyResource()
    assert my_resource._is_dagster_maintained()