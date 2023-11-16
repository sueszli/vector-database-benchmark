import re
import string
import typing
import pytest
from dagster import Any, ConfigMapping, DagsterInvalidConfigDefinitionError, DagsterInvalidConfigError, DagsterInvalidDefinitionError, Enum, EnumValue, Field, Float, GraphDefinition, Int, List, Noneable, Permissive, ResourceDefinition, Set, String, Tuple, graph, job, op
from dagster._check import ParameterCheckError
from dagster._config import DagsterEvaluationErrorReason, Map, Shape, convert_potential_field, process_config, validate_config
from dagster._utils.test import wrap_op_in_graph_and_execute

def test_noop_config():
    if False:
        while True:
            i = 10
    assert Field(Any)

def test_int_field():
    if False:
        while True:
            i = 10
    config_field = convert_potential_field({'int_field': Int})
    assert validate_config(config_field.config_type, {'int_field': 1}).value == {'int_field': 1}

def test_float_field():
    if False:
        print('Hello World!')
    config_field = convert_potential_field({'float_field': Float})
    assert validate_config(config_field.config_type, {'float_field': 1.0}).value == {'float_field': 1.0}
    assert process_config(config_field.config_type, {'float_field': 1.0}).value == {'float_field': 1.0}
    assert validate_config(config_field.config_type, {'float_field': 1}).value == {'float_field': 1}
    assert process_config(config_field.config_type, {'float_field': 1}).value == {'float_field': 1.0}

def assert_config_value_success(config_type, config_value, expected):
    if False:
        print('Hello World!')
    result = process_config(config_type, config_value)
    assert result.success
    assert result.value == expected

def assert_eval_failure(config_type, value):
    if False:
        print('Hello World!')
    assert not validate_config(config_type, value).success

def test_int_fails():
    if False:
        for i in range(10):
            print('nop')
    config_field = convert_potential_field({'int_field': Int})
    assert_eval_failure(config_field.config_type, {'int_field': 'fjkdj'})
    assert_eval_failure(config_field.config_type, {'int_field': True})

def test_default_arg():
    if False:
        for i in range(10):
            print('nop')
    config_field = convert_potential_field({'int_field': Field(Int, default_value=2, is_required=False)})
    assert_config_value_success(config_field.config_type, {}, {'int_field': 2})

def test_default_float_arg():
    if False:
        for i in range(10):
            print('nop')
    config_field = convert_potential_field({'float_field': Field(Float, default_value=2.0, is_required=False)})
    assert_config_value_success(config_field.config_type, {}, {'float_field': 2.0})
    config_field = convert_potential_field({'float_field': Field(Float, default_value=2, is_required=False)})
    assert_config_value_success(config_field.config_type, {}, {'float_field': 2})

def _single_required_enum_config_dict():
    if False:
        while True:
            i = 10
    return convert_potential_field({'enum_field': Enum('MyEnum', [EnumValue('OptionA'), EnumValue('OptionB')])})

def _single_required_string_config_dict():
    if False:
        print('Hello World!')
    return convert_potential_field({'string_field': String})

def _multiple_required_fields_config_dict():
    if False:
        return 10
    return convert_potential_field({'field_one': String, 'field_two': String})

def _single_optional_string_config_dict():
    if False:
        for i in range(10):
            print('nop')
    return convert_potential_field({'optional_field': Field(String, is_required=False)})

def _single_optional_string_field_config_dict_with_default():
    if False:
        return 10
    optional_field_def = Field(String, is_required=False, default_value='some_default')
    return convert_potential_field({'optional_field': optional_field_def})

def _mixed_required_optional_string_config_dict_with_default():
    if False:
        for i in range(10):
            print('nop')
    return convert_potential_field({'optional_arg': Field(String, is_required=False, default_value='some_default'), 'required_arg': Field(String, is_required=True), 'optional_arg_no_default': Field(String, is_required=False)})

def _multiple_required_fields_config_permissive_dict():
    if False:
        while True:
            i = 10
    return Field(Permissive({'field_one': Field(String), 'field_two': Field(String)}))

def _validate(config_field, value):
    if False:
        for i in range(10):
            print('nop')
    res = process_config(config_field.config_type, value)
    assert res.success, res.errors[0].message
    return res.value

def test_single_required_enum_field_config_type():
    if False:
        i = 10
        return i + 15
    assert _validate(_single_required_enum_config_dict(), {'enum_field': 'OptionA'}) == {'enum_field': 'OptionA'}
    expected_suggested_config = {'enum_field': 'OptionA'}
    with pytest.raises(AssertionError, match=f'Missing required config entry "enum_field" at the root. .* {expected_suggested_config}'):
        _validate(_single_required_enum_config_dict(), {})

def test_single_required_string_field_config_type():
    if False:
        i = 10
        return i + 15
    assert _validate(_single_required_string_config_dict(), {'string_field': 'value'}) == {'string_field': 'value'}
    with pytest.raises(AssertionError, match='Missing required config entry "string_field" at the root.'):
        _validate(_single_required_string_config_dict(), {})
    with pytest.raises(AssertionError):
        _validate(_single_required_string_config_dict(), {'extra': 'yup'})
    with pytest.raises(AssertionError):
        _validate(_single_required_string_config_dict(), {'string_field': 'yupup', 'extra': 'yup'})
    with pytest.raises(AssertionError):
        _validate(_single_required_string_config_dict(), {'string_field': 1})

def test_undefined_field_error():
    if False:
        print('Hello World!')
    with pytest.raises(AssertionError, match='Received unexpected config entry "extra" at the root. Expected: "{ string_field: String }".'):
        _validate(_single_required_string_config_dict(), {'string_field': 'value', 'extra': 'extra'})

def test_multiple_required_fields_passing():
    if False:
        for i in range(10):
            print('nop')
    assert _validate(_multiple_required_fields_config_dict(), {'field_one': 'value_one', 'field_two': 'value_two'}) == {'field_one': 'value_one', 'field_two': 'value_two'}

def test_multiple_required_fields_failing():
    if False:
        i = 10
        return i + 15
    expected_suggested_config = {'field_one': '...', 'field_two': '...'}
    with pytest.raises(AssertionError, match=f"Missing required config entries \\['field_one', 'field_two'\\] at the root. .* {expected_suggested_config}"):
        _validate(_multiple_required_fields_config_dict(), {})
    expected_suggested_config = {'field_two': '...'}
    with pytest.raises(AssertionError, match=f'Missing required config entry "field_two" at the root. .* {expected_suggested_config}'):
        _validate(_multiple_required_fields_config_dict(), {'field_one': 'yup'})
    with pytest.raises(AssertionError):
        _validate(_multiple_required_fields_config_dict(), {'field_one': 'yup', 'extra': 'yup'})
    with pytest.raises(AssertionError):
        _validate(_multiple_required_fields_config_dict(), {'field_one': 'yup', 'field_two': 'yup', 'extra': 'should_not_exist'})
    with pytest.raises(AssertionError):
        _validate(_multiple_required_fields_config_dict(), {'field_one': 'value_one', 'field_two': 2})

def test_single_optional_field_passing():
    if False:
        for i in range(10):
            print('nop')
    assert _validate(_single_optional_string_config_dict(), {'optional_field': 'value'}) == {'optional_field': 'value'}
    assert _validate(_single_optional_string_config_dict(), {}) == {}
    with pytest.raises(AssertionError):
        assert _validate(_single_optional_string_config_dict(), {'optional_field': None}) == {'optional_field': None}

def test_single_optional_field_failing():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(AssertionError):
        _validate(_single_optional_string_config_dict(), {'optional_field': 1})
    with pytest.raises(AssertionError):
        _validate(_single_optional_string_config_dict(), {'dlkjfalksdjflksaj': 1})

def test_single_optional_field_passing_with_default():
    if False:
        for i in range(10):
            print('nop')
    assert _validate(_single_optional_string_field_config_dict_with_default(), {}) == {'optional_field': 'some_default'}
    assert _validate(_single_optional_string_field_config_dict_with_default(), {'optional_field': 'override'}) == {'optional_field': 'override'}

def test_permissive_multiple_required_fields_passing():
    if False:
        for i in range(10):
            print('nop')
    assert _validate(_multiple_required_fields_config_permissive_dict(), {'field_one': 'value_one', 'field_two': 'value_two', 'previously_unspecified': 'should_exist'}) == {'field_one': 'value_one', 'field_two': 'value_two', 'previously_unspecified': 'should_exist'}

def test_permissive_multiple_required_fields_nested_passing():
    if False:
        i = 10
        return i + 15
    assert _validate(_multiple_required_fields_config_permissive_dict(), {'field_one': 'value_one', 'field_two': 'value_two', 'previously_unspecified': {'nested': 'value', 'with_int': 2}}) == {'field_one': 'value_one', 'field_two': 'value_two', 'previously_unspecified': {'nested': 'value', 'with_int': 2}}

def test_permissive_multiple_required_fields_failing():
    if False:
        while True:
            i = 10
    with pytest.raises(AssertionError):
        _validate(_multiple_required_fields_config_permissive_dict(), {})
    with pytest.raises(AssertionError):
        _validate(_multiple_required_fields_config_permissive_dict(), {'field_one': 'yup'})
    with pytest.raises(AssertionError):
        _validate(_multiple_required_fields_config_permissive_dict(), {'field_one': 'value_one', 'field_two': 2})

def test_map_passing():
    if False:
        i = 10
        return i + 15
    assert _validate(Field(Map(key_type=str, inner_type=str)), {'field_one': 'value_one', 'field_two': 'value_two'}) == {'field_one': 'value_one', 'field_two': 'value_two'}
    assert _validate(Field(Map(key_type=int, inner_type=float)), {5: 5.5, 3: 3.5}) == {5: 5.5, 3: 3.5}
    assert _validate(Field({str: int}), {'field_one': 2, 'field_two': 5}) == {'field_one': 2, 'field_two': 5}

def test_map_failing():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(ParameterCheckError):
        _validate(Field(Map(key_type='asdf', inner_type=str)), {'field_one': 'value_one', 'field_two': 2})
    with pytest.raises(ParameterCheckError) as e:
        _validate(Field(Map(Noneable(str), str)), {'field_one': 'value_one', 'field_two': 2})
    assert 'must be a scalar' in str(e)
    with pytest.raises(DagsterInvalidDefinitionError) as e:
        _validate(Field({55: str}), {'field_one': 'value_one', 'field_two': 2})
    assert 'Invalid key' in str(e)
    with pytest.raises(DagsterInvalidDefinitionError) as e:
        _validate(Field({Noneable(str): str}), {'field_one': 'value_one', 'field_two': 2})
    assert 'Non-scalar key' in str(e)
    with pytest.raises(AssertionError):
        _validate(Field(Map(key_type=str, inner_type=str)), {'field_one': 'value_one', 'field_two': 2})

def test_map_shape_complex():
    if False:
        for i in range(10):
            print('nop')
    assert _validate(Field(Map(str, Shape({'name': Field(str), 'number': Field(int)}))), {'foo': {'name': 'test_name', 'number': 5}, 'bar': {'name': 'other_name', 'number': 10}}) == {'foo': {'name': 'test_name', 'number': 5}, 'bar': {'name': 'other_name', 'number': 10}}
    assert _validate(Field({str: {'name': Field(str), 'number': Field(int)}}), {'foo': {'name': 'test_name', 'number': 5}, 'bar': {'name': 'other_name', 'number': 10}}) == {'foo': {'name': 'test_name', 'number': 5}, 'bar': {'name': 'other_name', 'number': 10}}
    with pytest.raises(AssertionError):
        _validate(Field(Map(str, Shape({'name': Field(str), 'number': Field(int)}))), {'foo': {'name': 'test_name', 'number': 'not_a_number'}, 'bar': {'name': 'other_name', 'number': 10}})
    with pytest.raises(AssertionError):
        _validate(Field(Map(str, Shape({'name': Field(str), 'number': Field(int)}))), {'foo': {'name': 'test_name', 'number': 15}, 'baz': 'not_a_shape'})

def test_mixed_args_passing():
    if False:
        while True:
            i = 10
    assert _validate(_mixed_required_optional_string_config_dict_with_default(), {'optional_arg': 'value_one', 'required_arg': 'value_two'}) == {'optional_arg': 'value_one', 'required_arg': 'value_two'}
    assert _validate(_mixed_required_optional_string_config_dict_with_default(), {'required_arg': 'value_two'}) == {'optional_arg': 'some_default', 'required_arg': 'value_two'}
    assert _validate(_mixed_required_optional_string_config_dict_with_default(), {'required_arg': 'value_two', 'optional_arg_no_default': 'value_three'}) == {'optional_arg': 'some_default', 'required_arg': 'value_two', 'optional_arg_no_default': 'value_three'}

def _single_nested_config():
    if False:
        return 10
    return convert_potential_field({'nested': {'int_field': Int}})

def _nested_optional_config_with_default():
    if False:
        i = 10
        return i + 15
    return convert_potential_field({'nested': {'int_field': Field(Int, is_required=False, default_value=3)}})

def _nested_optional_config_with_no_default():
    if False:
        print('Hello World!')
    return convert_potential_field({'nested': {'int_field': Field(Int, is_required=False)}})

def test_single_nested_config():
    if False:
        return 10
    assert _validate(_single_nested_config(), {'nested': {'int_field': 2}}) == {'nested': {'int_field': 2}}

def test_single_nested_config_undefined_errors():
    if False:
        print('Hello World!')
    with pytest.raises(AssertionError, match='Value at path root:nested must be dict. Expected: "{ int_field: Int }".'):
        _validate(_single_nested_config(), {'nested': 'dkjfdk'})
    with pytest.raises(AssertionError, match='Invalid scalar at path root:nested:int_field. Value "dkjfdk" of type .* is not valid for expected type "Int".'):
        _validate(_single_nested_config(), {'nested': {'int_field': 'dkjfdk'}})
    with pytest.raises(AssertionError, match='Received unexpected config entry "not_a_field" at path root:nested. Expected: "{ int_field: Int }".'):
        _validate(_single_nested_config(), {'nested': {'int_field': 2, 'not_a_field': 1}})
    with pytest.raises(AssertionError, match='Invalid scalar at path root:nested:int_field. Value "{\'too_nested\': \'dkjfdk\'}" of type .* is not valid for expected type "Int".'):
        _validate(_single_nested_config(), {'nested': {'int_field': {'too_nested': 'dkjfdk'}}})

def test_nested_optional_with_default():
    if False:
        for i in range(10):
            print('nop')
    assert _validate(_nested_optional_config_with_default(), {'nested': {'int_field': 2}}) == {'nested': {'int_field': 2}}
    assert _validate(_nested_optional_config_with_default(), {'nested': {}}) == {'nested': {'int_field': 3}}

def test_nested_optional_with_no_default():
    if False:
        return 10
    assert _validate(_nested_optional_config_with_no_default(), {'nested': {'int_field': 2}}) == {'nested': {'int_field': 2}}
    assert _validate(_nested_optional_config_with_no_default(), {'nested': {}}) == {'nested': {}}

def test_config_defaults():
    if False:
        return 10

    @op(config_schema={'sum': Int})
    def two(_context):
        if False:
            for i in range(10):
                print('nop')
        assert _context.op_config['sum'] == 6
        return _context.op_config['sum']

    @op(config_schema={'sum': Int})
    def one(_context, prev_sum):
        if False:
            print('Hello World!')
        assert prev_sum == 6
        return prev_sum + _context.op_config['sum']

    def addition_graph_config_fn(config):
        if False:
            for i in range(10):
                print('nop')
        child_config = {'config': {'sum': config['a'] + config['b'] + config['c']}}
        return {'one': child_config, 'two': child_config}

    @graph(config=ConfigMapping(config_schema={'a': Field(Int, is_required=False, default_value=1), 'b': Field(Int, is_required=False, default_value=2), 'c': Int}, config_fn=addition_graph_config_fn))
    def addition_graph():
        if False:
            while True:
                i = 10
        return one(two())

    @job
    def addition_job():
        if False:
            for i in range(10):
                print('nop')
        addition_graph()
    result = addition_job.execute_in_process({'ops': {'addition_graph': {'config': {'c': 3}}}})
    assert result.success

def test_config_with_and_without_config():
    if False:
        for i in range(10):
            print('nop')

    @op(config_schema={'prefix': Field(str, is_required=False, default_value='_')})
    def prefix_value(context, v):
        if False:
            while True:
                i = 10
        return '{prefix}{v}'.format(prefix=context.op_config['prefix'], v=v)

    @graph(config=ConfigMapping(config_schema={'prefix': Field(str, is_required=False, default_value='_id_')}, config_fn=lambda cfg: {'prefix_value': {'config': {'prefix': cfg['prefix']}}}))
    def prefix_id(val):
        if False:
            print('Hello World!')
        return prefix_value(val)

    @op
    def print_value(_, v):
        if False:
            i = 10
            return i + 15
        return str(v)

    @job
    def config_issue_job():
        if False:
            print('Hello World!')
        v = prefix_id()
        print_value(v)
    result = config_issue_job.execute_in_process({'ops': {'prefix_id': {'config': {'prefix': '_customprefix_'}, 'inputs': {'val': {'value': '12345'}}}}})
    assert result.success
    assert result.output_for_node('print_value') == '_customprefix_12345'
    result_using_default = config_issue_job.execute_in_process({'ops': {'prefix_id': {'config': {}, 'inputs': {'val': {'value': '12345'}}}}})
    assert result_using_default.success
    assert result_using_default.output_for_node('print_value') == '_id_12345'

def test_build_optionality():
    if False:
        while True:
            i = 10
    optional_test_type = convert_potential_field({'required': {'value': String}, 'optional': {'value': Field(String, is_required=False)}}).config_type
    assert optional_test_type.fields['required'].is_required
    assert optional_test_type.fields['optional'].is_required is False

def test_wrong_op_name():
    if False:
        return 10

    @op(name='some_op', ins={}, out={}, config_schema=Int)
    def some_op(_):
        if False:
            print('Hello World!')
        return None

    @job(name='job_wrong_op_name')
    def job_def():
        if False:
            while True:
                i = 10
        some_op()
    env_config = {'ops': {'another_name': {'config': {}}}}
    with pytest.raises(DagsterInvalidConfigError) as pe_info:
        job_def.execute_in_process(env_config)
    pe = pe_info.value
    assert 'Received unexpected config entry "another_name" at path root:ops' in str(pe)

def fail_me():
    if False:
        while True:
            i = 10
    assert False

def dummy_resource(config_schema=None):
    if False:
        while True:
            i = 10
    return ResourceDefinition(lambda _: None, config_schema=config_schema)

def test_wrong_resources():
    if False:
        while True:
            i = 10
    job_def = GraphDefinition(name='job_test_multiple_context', node_defs=[]).to_job(resource_defs={'resource_one': dummy_resource(), 'resource_two': dummy_resource()})
    with pytest.raises(DagsterInvalidConfigError, match='Received unexpected config entry "nope" at path root:resources'):
        job_def.execute_in_process({'resources': {'nope': {}}})

def test_op_list_config():
    if False:
        for i in range(10):
            print('nop')
    value = [1, 2]
    called = {}

    @op(name='op_list_config', ins={}, out={}, config_schema=[int])
    def op_list_config(context):
        if False:
            return 10
        assert context.op_config == value
        called['yup'] = True

    @job(name='op_list_config_job')
    def job_def():
        if False:
            return 10
        op_list_config()
    result = job_def.execute_in_process(run_config={'ops': {'op_list_config': {'config': value}}})
    assert result.success
    assert called['yup']

def test_two_list_types():
    if False:
        for i in range(10):
            print('nop')

    @op(ins={}, config_schema={'list_one': [int], 'list_two': [int]})
    def two_list_type(context):
        if False:
            while True:
                i = 10
        return context.op_config
    assert wrap_op_in_graph_and_execute(two_list_type, run_config={'ops': {'two_list_type': {'config': {'list_one': [1], 'list_two': [2]}}}}).output_value() == {'list_one': [1], 'list_two': [2]}

    @op(ins={}, config_schema={'list_one': [Int], 'list_two': [Int]})
    def two_list_type_condensed_syntax(context):
        if False:
            for i in range(10):
                print('nop')
        return context.op_config
    assert wrap_op_in_graph_and_execute(two_list_type_condensed_syntax, run_config={'ops': {'two_list_type_condensed_syntax': {'config': {'list_one': [1], 'list_two': [2]}}}}).output_value() == {'list_one': [1], 'list_two': [2]}

    @op(ins={}, config_schema={'list_one': [int], 'list_two': [int]})
    def two_list_type_condensed_syntax_primitives(context):
        if False:
            i = 10
            return i + 15
        return context.op_config
    assert wrap_op_in_graph_and_execute(two_list_type_condensed_syntax_primitives, run_config={'ops': {'two_list_type_condensed_syntax_primitives': {'config': {'list_one': [1], 'list_two': [2]}}}}).output_value() == {'list_one': [1], 'list_two': [2]}

def test_multilevel_default_handling():
    if False:
        return 10

    @op(config_schema=Field(Int, is_required=False, default_value=234))
    def has_default_value(context):
        if False:
            i = 10
            return i + 15
        assert context.op_config == 234
    job_def = GraphDefinition(name='multilevel_default_handling', node_defs=[has_default_value]).to_job()
    assert job_def.execute_in_process().success
    assert job_def.execute_in_process(run_config=None).success
    assert job_def.execute_in_process(run_config={}).success
    assert job_def.execute_in_process(run_config={'ops': {}}).success
    assert job_def.execute_in_process(run_config={'ops': {'has_default_value': {}}}).success
    assert job_def.execute_in_process(run_config={'ops': {'has_default_value': {'config': 234}}}).success

def test_no_env_missing_required_error_handling():
    if False:
        while True:
            i = 10

    @op(config_schema=Int)
    def required_int_op(_context):
        if False:
            return 10
        pass
    job_def = GraphDefinition(name='no_env_missing_required_error', node_defs=[required_int_op]).to_job()
    with pytest.raises(DagsterInvalidConfigError) as pe_info:
        job_def.execute_in_process()
    assert isinstance(pe_info.value, DagsterInvalidConfigError)
    pe = pe_info.value
    assert len(pe.errors) == 1
    mfe = pe.errors[0]
    assert mfe.reason == DagsterEvaluationErrorReason.MISSING_REQUIRED_FIELD
    assert len(pe.errors) == 1
    expected_suggested_config = {'ops': {'required_int_op': {'config': 0}}}
    assert pe.errors[0].message.startswith('Missing required config entry "ops" at the root.')
    assert str(expected_suggested_config) in pe.errors[0].message

def test_root_extra_field():
    if False:
        i = 10
        return i + 15

    @op(config_schema=Int)
    def required_int_op(_context):
        if False:
            i = 10
            return i + 15
        pass

    @job
    def job_def():
        if False:
            i = 10
            return i + 15
        required_int_op()
    with pytest.raises(DagsterInvalidConfigError) as pe_info:
        job_def.execute_in_process(run_config={'ops': {'required_int_op': {'config': 948594}}, 'nope': None})
    pe = pe_info.value
    assert len(pe.errors) == 1
    fnd = pe.errors[0]
    assert fnd.reason == DagsterEvaluationErrorReason.FIELD_NOT_DEFINED
    assert 'Received unexpected config entry "nope"' in pe.message

def test_deeper_path():
    if False:
        print('Hello World!')

    @op(config_schema=Int)
    def required_int_op(_context):
        if False:
            return 10
        pass

    @job
    def job_def():
        if False:
            while True:
                i = 10
        required_int_op()
    with pytest.raises(DagsterInvalidConfigError) as pe_info:
        job_def.execute_in_process(run_config={'ops': {'required_int_op': {'config': 'asdf'}}})
    pe = pe_info.value
    assert len(pe.errors) == 1
    rtm = pe.errors[0]
    assert rtm.reason == DagsterEvaluationErrorReason.RUNTIME_TYPE_MISMATCH

def test_working_list_path():
    if False:
        return 10
    called = {}

    @op(config_schema=[int])
    def required_list_int_op(context):
        if False:
            for i in range(10):
                print('nop')
        assert context.op_config == [1, 2]
        called['yup'] = True

    @job
    def job_def():
        if False:
            return 10
        required_list_int_op()
    result = job_def.execute_in_process(run_config={'ops': {'required_list_int_op': {'config': [1, 2]}}})
    assert result.success
    assert called['yup']

def test_item_error_list_path():
    if False:
        while True:
            i = 10
    called = {}

    @op(config_schema=[int])
    def required_list_int_op(context):
        if False:
            print('Hello World!')
        assert context.op_config == [1, 2]
        called['yup'] = True

    @job
    def job_def():
        if False:
            for i in range(10):
                print('nop')
        required_list_int_op()
    with pytest.raises(DagsterInvalidConfigError) as pe_info:
        job_def.execute_in_process(run_config={'ops': {'required_list_int_op': {'config': [1, 'nope']}}})
    pe = pe_info.value
    assert len(pe.errors) == 1
    rtm = pe.errors[0]
    assert rtm.reason == DagsterEvaluationErrorReason.RUNTIME_TYPE_MISMATCH
    assert 'Invalid scalar at path root:ops:required_list_int_op:config[1]' in str(pe)

def test_list_in_config_error():
    if False:
        print('Hello World!')
    error_msg = 'Cannot use List in the context of config. Please use a python list (e.g. [int]) or dagster.Array (e.g. Array(int)) instead.'
    with pytest.raises(DagsterInvalidDefinitionError, match=re.escape(error_msg)):

        @op(config_schema=List[int])
        def _no_runtime_list_in_config(_):
            if False:
                i = 10
                return i + 15
            pass

def test_working_map_path():
    if False:
        while True:
            i = 10
    called = {}

    @op(config_schema={str: int})
    def required_map_int_op(context):
        if False:
            print('Hello World!')
        assert context.op_config == {'foo': 1, 'bar': 2}
        called['yup'] = True

    @job
    def job_def():
        if False:
            while True:
                i = 10
        required_map_int_op()
    result = job_def.execute_in_process(run_config={'ops': {'required_map_int_op': {'config': {'foo': 1, 'bar': 2}}}})
    assert result.success
    assert called['yup']

def test_item_error_map_path():
    if False:
        print('Hello World!')
    called = {}

    @op(config_schema={str: int})
    def required_map_int_op(context):
        if False:
            i = 10
            return i + 15
        assert context.op_config == {'foo': 1, 'bar': 2}
        called['yup'] = True

    @job
    def job_def():
        if False:
            print('Hello World!')
        required_map_int_op()
    with pytest.raises(DagsterInvalidConfigError) as pe_info:
        job_def.execute_in_process(run_config={'ops': {'required_map_int_op': {'config': {'foo': 1, 'bar': 'nope'}}}})
    pe = pe_info.value
    assert len(pe.errors) == 1
    rtm = pe.errors[0]
    assert rtm.reason == DagsterEvaluationErrorReason.RUNTIME_TYPE_MISMATCH
    assert "Invalid scalar at path root:ops:required_map_int_op:config:'bar'" in str(pe)

def test_required_resource_not_given():
    if False:
        return 10

    @op(required_resource_keys={'required'})
    def needs_resource(_):
        if False:
            print('Hello World!')
        pass

    @job(name='required_resource_not_given', resource_defs={'required': dummy_resource(Int)})
    def job_def():
        if False:
            while True:
                i = 10
        needs_resource()
    with pytest.raises(DagsterInvalidConfigError) as not_none_pe_info:
        job_def.execute_in_process(run_config={'resources': None})
    assert len(not_none_pe_info.value.errors) == 1
    assert 'Value at path root:resources must not be None.' in not_none_pe_info.value.errors[0].message
    with pytest.raises(DagsterInvalidConfigError) as pe_info:
        job_def.execute_in_process(run_config={'resources': {}})
    pe = pe_info.value
    error = pe.errors[0]
    assert error.reason == DagsterEvaluationErrorReason.MISSING_REQUIRED_FIELD
    expected_suggested_config = {'required': {'config': 0}}
    assert error.message.startswith('Missing required config entry "required" at path root:resources.')
    assert str(expected_suggested_config) in error.message

def test_multilevel_good_error_handling_ops():
    if False:
        for i in range(10):
            print('nop')

    @op(config_schema=Int)
    def good_error_handling(_context):
        if False:
            print('Hello World!')
        pass

    @job
    def job_def():
        if False:
            return 10
        good_error_handling()
    with pytest.raises(DagsterInvalidConfigError) as not_none_pe_info:
        job_def.execute_in_process(run_config={'ops': None})
    assert len(not_none_pe_info.value.errors) == 1
    assert 'Value at path root:ops must not be None.' in not_none_pe_info.value.errors[0].message
    with pytest.raises(DagsterInvalidConfigError) as missing_field_pe_info:
        job_def.execute_in_process(run_config={'ops': {}})
    assert len(missing_field_pe_info.value.errors) == 1
    expected_suggested_config = {'good_error_handling': {'config': 0}}
    assert missing_field_pe_info.value.errors[0].message.startswith('Missing required config entry "good_error_handling" at path root:ops.')
    assert str(expected_suggested_config) in missing_field_pe_info.value.errors[0].message

def test_multilevel_good_error_handling_op_name_ops():
    if False:
        return 10

    @op(config_schema=Int)
    def good_error_handling(_context):
        if False:
            for i in range(10):
                print('nop')
        pass

    @job
    def job_def():
        if False:
            while True:
                i = 10
        good_error_handling()
    with pytest.raises(DagsterInvalidConfigError) as pe_info:
        job_def.execute_in_process(run_config={'ops': {'good_error_handling': {}}})
    assert len(pe_info.value.errors) == 1
    expected_suggested_config = {'config': 0}
    assert pe_info.value.errors[0].message.startswith('Missing required config entry "config" at path root:ops:good_error_handling.')
    assert str(expected_suggested_config) in pe_info.value.errors[0].message

def test_multilevel_good_error_handling_config_ops_name_ops():
    if False:
        return 10

    @op(config_schema=Noneable(int))
    def good_error_handling(_context):
        if False:
            while True:
                i = 10
        pass

    @job
    def job_def():
        if False:
            i = 10
            return i + 15
        good_error_handling()
    job_def.execute_in_process(run_config={'ops': {'good_error_handling': {'config': None}}})

def test_invalid_default_values():
    if False:
        while True:
            i = 10
    with pytest.raises(DagsterInvalidConfigError, match='Value "3" of type .* is not valid for expected type "Int"'):

        @op(config_schema=Field(Int, default_value='3'))
        def _op(_):
            if False:
                for i in range(10):
                    print('nop')
            pass

def test_typing_types_into_config():
    if False:
        i = 10
        return i + 15
    match_str = re.escape('You have passed in typing.List to the config system. Types from the typing module in python are not allowed in the config system. You must use types that are imported from dagster or primitive types such as bool, int, etc.')
    with pytest.raises(DagsterInvalidDefinitionError, match=match_str):

        @op(config_schema=Field(typing.List))
        def _op(_):
            if False:
                while True:
                    i = 10
            pass
    with pytest.raises(DagsterInvalidDefinitionError, match=match_str):

        @op(config_schema=typing.List)
        def _op(_):
            if False:
                i = 10
                return i + 15
            pass
    match_str = re.escape('You have passed in typing.List[int] to the config system. Types from the typing module in python are not allowed in the config system. You must use types that are imported from dagster or primitive types such as bool, int, etc.')
    with pytest.raises(DagsterInvalidDefinitionError, match=match_str):

        @op(config_schema=Field(typing.List[int]))
        def _op(_):
            if False:
                i = 10
                return i + 15
            pass
    with pytest.raises(DagsterInvalidDefinitionError, match=match_str):

        @op(config_schema=typing.List[int])
        def _op(_):
            if False:
                for i in range(10):
                    print('nop')
            pass
    for ttype in [typing.Optional[int], typing.Set, typing.Set[int], typing.Dict, typing.Dict[int, str], typing.Tuple, typing.Tuple[int, int]]:
        with pytest.raises(DagsterInvalidDefinitionError):

            @op(config_schema=Field(ttype))
            def _op(_):
                if False:
                    while True:
                        i = 10
                pass

def test_no_set_in_config_system():
    if False:
        i = 10
        return i + 15
    set_error_msg = re.escape('Cannot use Set in the context of a config field.')
    with pytest.raises(DagsterInvalidDefinitionError, match=set_error_msg):

        @op(config_schema=Field(Set))
        def _bare_open_set(_):
            if False:
                return 10
            pass
    with pytest.raises(DagsterInvalidDefinitionError, match=set_error_msg):

        @op(config_schema=Set)
        def _bare_open_set(_):
            if False:
                for i in range(10):
                    print('nop')
            pass
    with pytest.raises(DagsterInvalidDefinitionError, match=set_error_msg):

        @op(config_schema=Field(Set[int]))
        def _bare_closed_set(_):
            if False:
                i = 10
                return i + 15
            pass
    with pytest.raises(DagsterInvalidDefinitionError, match=set_error_msg):

        @op(config_schema=Set[int])
        def _bare_closed_set(_):
            if False:
                for i in range(10):
                    print('nop')
            pass

def test_no_tuple_in_config_system():
    if False:
        for i in range(10):
            print('nop')
    tuple_error_msg = re.escape('Cannot use Tuple in the context of a config field.')
    with pytest.raises(DagsterInvalidDefinitionError, match=tuple_error_msg):

        @op(config_schema=Field(Tuple))
        def _bare_open_tuple(_):
            if False:
                print('Hello World!')
            pass
    with pytest.raises(DagsterInvalidDefinitionError, match=tuple_error_msg):

        @op(config_schema=Field(Tuple[int]))
        def _bare_closed_set(_):
            if False:
                print('Hello World!')
            pass

def test_field_is_none():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(DagsterInvalidConfigDefinitionError) as exc_info:

        @op(config_schema={'none_field': None})
        def _none_is_bad(_):
            if False:
                for i in range(10):
                    print('nop')
            pass
    assert 'Fields cannot be None' in str(exc_info.value)

def test_permissive_defaults():
    if False:
        for i in range(10):
            print('nop')

    @op(config_schema=Permissive({'four': Field(int, default_value=4)}))
    def perm_with_defaults(context):
        if False:
            print('Hello World!')
        assert context.op_config['four'] == 4
    assert wrap_op_in_graph_and_execute(perm_with_defaults).success

def test_permissive_ordering():
    if False:
        i = 10
        return i + 15
    alphabet = {letter: letter for letter in string.ascii_lowercase}

    @op(config_schema=dict)
    def test_order(context):
        if False:
            for i in range(10):
                print('nop')
        alpha = list(context.op_config.keys())
        for (idx, letter) in enumerate(string.ascii_lowercase):
            assert letter == alpha[idx]
    assert wrap_op_in_graph_and_execute(test_order, run_config={'ops': {'test_order': {'config': alphabet}}}).success