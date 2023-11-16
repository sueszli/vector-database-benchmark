import pickle
import pytest
from dagster import Any, Bool, DagsterInvalidConfigError, Float, GraphDefinition, In, Int, List, Optional, Out, String, op
from dagster._utils.test import get_temp_file_name

def _execute_job_with_subset(job_def, run_config, op_selection):
    if False:
        for i in range(10):
            print('nop')
    return job_def.get_subset(op_selection=op_selection).execute_in_process(run_config=run_config)

def define_test_all_scalars_job():
    if False:
        i = 10
        return i + 15

    @op(ins={'num': In(Int)})
    def take_int(num):
        if False:
            print('Hello World!')
        return num

    @op(out=Out(Int))
    def produce_int():
        if False:
            while True:
                i = 10
        return 2

    @op(ins={'string': In(String)})
    def take_string(string):
        if False:
            while True:
                i = 10
        return string

    @op(out=Out(String))
    def produce_string():
        if False:
            while True:
                i = 10
        return 'foo'

    @op(ins={'float_number': In(Float)})
    def take_float(float_number):
        if False:
            while True:
                i = 10
        return float_number

    @op(out=Out(Float))
    def produce_float():
        if False:
            print('Hello World!')
        return 3.14

    @op(ins={'bool_value': In(Bool)})
    def take_bool(bool_value):
        if False:
            while True:
                i = 10
        return bool_value

    @op(out=Out(Bool))
    def produce_bool():
        if False:
            print('Hello World!')
        return True

    @op(ins={'any_value': In(Any)})
    def take_any(any_value):
        if False:
            return 10
        return any_value

    @op(out=Out(Any))
    def produce_any():
        if False:
            while True:
                i = 10
        return True

    @op(ins={'string_list': In(List[String])})
    def take_string_list(string_list):
        if False:
            i = 10
            return i + 15
        return string_list

    @op(ins={'nullable_string': In(Optional[String])})
    def take_nullable_string(nullable_string):
        if False:
            for i in range(10):
                print('nop')
        return nullable_string
    return GraphDefinition(name='test_all_scalars_job', node_defs=[produce_any, produce_bool, produce_float, produce_int, produce_string, take_any, take_bool, take_float, take_int, take_nullable_string, take_string, take_string_list]).to_job()

def single_input_env(solid_name, input_name, input_spec):
    if False:
        i = 10
        return i + 15
    return {'ops': {solid_name: {'inputs': {input_name: input_spec}}}}

def test_int_input_schema_value():
    if False:
        i = 10
        return i + 15
    result = _execute_job_with_subset(define_test_all_scalars_job(), run_config={'ops': {'take_int': {'inputs': {'num': {'value': 2}}}}}, op_selection=['take_int'])
    assert result.success
    assert result.output_for_node('take_int') == 2

def test_int_input_schema_raw_value():
    if False:
        print('Hello World!')
    result = _execute_job_with_subset(define_test_all_scalars_job(), run_config={'ops': {'take_int': {'inputs': {'num': 2}}}}, op_selection=['take_int'])
    assert result.success
    assert result.output_for_node('take_int') == 2

def test_int_input_schema_failure_wrong_value_type():
    if False:
        print('Hello World!')
    with pytest.raises(DagsterInvalidConfigError, match='Invalid scalar at path root:ops:take_int:inputs:num:value'):
        _execute_job_with_subset(define_test_all_scalars_job(), run_config=single_input_env('take_int', 'num', {'value': 'dkjdfkdj'}), op_selection=['take_int'])

def test_int_input_schema_failure_wrong_key():
    if False:
        print('Hello World!')
    with pytest.raises(DagsterInvalidConfigError) as exc_info:
        _execute_job_with_subset(define_test_all_scalars_job(), run_config=single_input_env('take_int', 'num', {'wrong_key': 'dkjdfkdj'}), op_selection=['take_int'])
    assert 'Error 1: Received unexpected config entry "wrong_key" at path root:ops:take_int:inputs:num' in str(exc_info.value)

def test_int_input_schema_failure_raw_string():
    if False:
        i = 10
        return i + 15
    with pytest.raises(DagsterInvalidConfigError) as exc_info:
        _execute_job_with_subset(define_test_all_scalars_job(), run_config=single_input_env('take_int', 'num', 'dkjdfkdj'), op_selection=['take_int'])
    assert 'Error 1: Invalid scalar at path root:ops:take_int:inputs:num' in str(exc_info.value)

def single_output_env(solid_name, output_spec):
    if False:
        i = 10
        return i + 15
    return {'ops': {solid_name: {'outputs': [{'result': output_spec}]}}}

def test_int_input_schema_json():
    if False:
        while True:
            i = 10
    with get_temp_file_name() as tmp_file:
        with open(tmp_file, 'w') as ff:
            ff.write('{"value": 2}')
        source_result = _execute_job_with_subset(define_test_all_scalars_job(), run_config=single_input_env('take_int', 'num', {'json': {'path': tmp_file}}), op_selection=['take_int'])
        assert source_result.output_for_node('take_int') == 2

def test_int_input_schema_pickle():
    if False:
        i = 10
        return i + 15
    with get_temp_file_name() as tmp_file:
        with open(tmp_file, 'wb') as ff:
            pickle.dump(2, ff)
        source_result = _execute_job_with_subset(define_test_all_scalars_job(), run_config=single_input_env('take_int', 'num', {'pickle': {'path': tmp_file}}), op_selection=['take_int'])
        assert source_result.output_for_node('take_int') == 2

def test_string_input_schema_value():
    if False:
        while True:
            i = 10
    result = _execute_job_with_subset(define_test_all_scalars_job(), run_config=single_input_env('take_string', 'string', {'value': 'dkjkfd'}), op_selection=['take_string'])
    assert result.success
    assert result.output_for_node('take_string') == 'dkjkfd'

def test_string_input_schema_failure():
    if False:
        i = 10
        return i + 15
    with pytest.raises(DagsterInvalidConfigError) as exc_info:
        _execute_job_with_subset(define_test_all_scalars_job(), run_config=single_input_env('take_string', 'string', {'value': 3343}), op_selection=['take_string'])
    assert 'Invalid scalar at path root:ops:take_string:inputs:string:value' in str(exc_info.value)

def test_float_input_schema_value():
    if False:
        for i in range(10):
            print('nop')
    result = _execute_job_with_subset(define_test_all_scalars_job(), run_config=single_input_env('take_float', 'float_number', {'value': 3.34}), op_selection=['take_float'])
    assert result.success
    assert result.output_for_node('take_float') == 3.34

def test_float_input_schema_failure():
    if False:
        return 10
    with pytest.raises(DagsterInvalidConfigError) as exc_info:
        _execute_job_with_subset(define_test_all_scalars_job(), run_config=single_input_env('take_float', 'float_number', {'value': '3343'}), op_selection=['take_float'])
    assert 'Invalid scalar at path root:ops:take_float:inputs:float_number:value' in str(exc_info.value)

def test_bool_input_schema_value():
    if False:
        print('Hello World!')
    result = _execute_job_with_subset(define_test_all_scalars_job(), run_config=single_input_env('take_bool', 'bool_value', {'value': True}), op_selection=['take_bool'])
    assert result.success
    assert result.output_for_node('take_bool') is True

def test_bool_input_schema_failure():
    if False:
        return 10
    with pytest.raises(DagsterInvalidConfigError) as exc_info:
        _execute_job_with_subset(define_test_all_scalars_job(), run_config=single_input_env('take_bool', 'bool_value', {'value': '3343'}), op_selection=['take_bool'])
    assert 'Invalid scalar at path root:ops:take_bool:inputs:bool_value:value' in str(exc_info.value)

def test_any_input_schema_value():
    if False:
        while True:
            i = 10
    result = _execute_job_with_subset(define_test_all_scalars_job(), run_config=single_input_env('take_any', 'any_value', {'value': 'ff'}), op_selection=['take_any'])
    assert result.success
    assert result.output_for_node('take_any') == 'ff'
    result = _execute_job_with_subset(define_test_all_scalars_job(), run_config=single_input_env('take_any', 'any_value', {'value': 3843}), op_selection=['take_any'])
    assert result.success
    assert result.output_for_node('take_any') == 3843

def test_none_string_input_schema_failure():
    if False:
        i = 10
        return i + 15
    with pytest.raises(DagsterInvalidConfigError) as exc_info:
        _execute_job_with_subset(define_test_all_scalars_job(), run_config=single_input_env('take_string', 'string', None), op_selection=['take_string'])
    assert len(exc_info.value.errors) == 1
    error = exc_info.value.errors[0]
    assert 'Value at path root:ops:take_string:inputs:string must not be None.' in error.message

def test_value_none_string_input_schema_failure():
    if False:
        return 10
    with pytest.raises(DagsterInvalidConfigError) as exc_info:
        _execute_job_with_subset(define_test_all_scalars_job(), run_config=single_input_env('take_string', 'string', {'value': None}), op_selection=['take_string'])
    assert 'Value at path root:ops:take_string:inputs:string:value must not be None' in str(exc_info.value)

def test_string_input_schema_json():
    if False:
        i = 10
        return i + 15
    with get_temp_file_name() as tmp_file:
        with open(tmp_file, 'w') as ff:
            ff.write('{"value": "foo"}')
        source_result = _execute_job_with_subset(define_test_all_scalars_job(), run_config=single_input_env('take_string', 'string', {'json': {'path': tmp_file}}), op_selection=['take_string'])
        assert source_result.output_for_node('take_string') == 'foo'

def test_string_input_schema_pickle():
    if False:
        print('Hello World!')
    with get_temp_file_name() as tmp_file:
        with open(tmp_file, 'wb') as ff:
            pickle.dump('foo', ff)
        source_result = _execute_job_with_subset(define_test_all_scalars_job(), run_config=single_input_env('take_string', 'string', {'pickle': {'path': tmp_file}}), op_selection=['take_string'])
        assert source_result.output_for_node('take_string') == 'foo'

def test_string_list_input():
    if False:
        return 10
    result = _execute_job_with_subset(define_test_all_scalars_job(), run_config=single_input_env('take_string_list', 'string_list', [{'value': 'foobar'}]), op_selection=['take_string_list'])
    assert result.success
    assert result.output_for_node('take_string_list') == ['foobar']

def test_nullable_string_input_with_value():
    if False:
        for i in range(10):
            print('nop')
    result = _execute_job_with_subset(define_test_all_scalars_job(), run_config=single_input_env('take_nullable_string', 'nullable_string', {'value': 'foobar'}), op_selection=['take_nullable_string'])
    assert result.success
    assert result.output_for_node('take_nullable_string') == 'foobar'

def test_nullable_string_input_with_none_value():
    if False:
        i = 10
        return i + 15
    with pytest.raises(DagsterInvalidConfigError) as exc_info:
        _execute_job_with_subset(define_test_all_scalars_job(), run_config=single_input_env('take_nullable_string', 'nullable_string', {'value': None}), op_selection=['take_nullable_string'])
    assert 'Value at path root:ops:take_nullable_string:inputs:nullable_string:value must not be None' in str(exc_info.value)

def test_nullable_string_input_without_value():
    if False:
        while True:
            i = 10
    result = _execute_job_with_subset(define_test_all_scalars_job(), run_config=single_input_env('take_nullable_string', 'nullable_string', None), op_selection=['take_nullable_string'])
    assert result.success
    assert result.output_for_node('take_nullable_string') is None