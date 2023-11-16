import pytest
from dagster import In, op
from dagster._core.errors import DagsterInvalidDefinitionError
from dagster._utils.test import wrap_op_in_graph_and_execute

def test_op_input_arguments():
    if False:
        for i in range(10):
            print('nop')

    @op
    def _no_param():
        if False:
            return 10
        pass

    @op
    def _underscore_param(_):
        if False:
            while True:
                i = 10
        pass
    assert '_' not in _underscore_param.input_dict

    @op
    def _context_param_underscore(_context):
        if False:
            i = 10
            return i + 15
        pass
    assert '_context' not in _context_param_underscore.input_dict

    @op
    def _context_param_back_underscore(context_):
        if False:
            i = 10
            return i + 15
        pass
    assert 'context_' not in _context_param_back_underscore.input_dict

    @op
    def _context_param_regular(context):
        if False:
            for i in range(10):
                print('nop')
        pass
    assert 'context' not in _context_param_regular.input_dict

    @op
    def _context_with_inferred_inputs(context, _x, _y):
        if False:
            for i in range(10):
                print('nop')
        pass
    assert '_x' in _context_with_inferred_inputs.input_dict
    assert '_y' in _context_with_inferred_inputs.input_dict
    assert 'context' not in _context_with_inferred_inputs.input_dict

    @op
    def _context_with_inferred_invalid_inputs(context, _context, context_):
        if False:
            return 10
        pass

    @op
    def _context_with_underscore_arg(context, _):
        if False:
            print('Hello World!')
        pass

    @op(ins={'x': In()})
    def _context_with_input_definitions(context, x):
        if False:
            while True:
                i = 10
        pass

    @op
    def _inputs_with_no_context(x, y):
        if False:
            print('Hello World!')
        pass
    with pytest.raises(DagsterInvalidDefinitionError, match='"context" is not a valid name in Dagster. It conflicts with a Dagster or python reserved keyword.'):

        @op
        def _context_after_inputs(x, context):
            if False:
                i = 10
                return i + 15
            pass

    @op(ins={'_': In()})
    def _underscore_after_input_arg(x, _):
        if False:
            return 10
        pass

    @op(ins={'_x': In()})
    def _context_partial_inputs(context, _x):
        if False:
            print('Hello World!')
        pass

    @op(ins={'x': In()})
    def _context_partial_inputs_2(x, y):
        if False:
            return 10
        pass

    @op
    def _context_arguments_out_of_order_still_works(_, x, _context):
        if False:
            while True:
                i = 10
        pass
    assert 'x' in _context_arguments_out_of_order_still_works.input_dict
    assert '_context' in _context_arguments_out_of_order_still_works.input_dict

def test_execution_cases():
    if False:
        return 10

    @op
    def underscore_inputs(x, _):
        if False:
            return 10
        return x + _
    assert wrap_op_in_graph_and_execute(underscore_inputs, input_values={'x': 5, '_': 6}).output_value() == 11

    @op
    def context_underscore_inputs(context, x, _):
        if False:
            return 10
        return x + _
    assert wrap_op_in_graph_and_execute(context_underscore_inputs, input_values={'x': 5, '_': 6}).output_value() == 11

    @op
    def underscore_context_poorly_named_input(_, x, context_):
        if False:
            print('Hello World!')
        return x + context_
    assert wrap_op_in_graph_and_execute(underscore_context_poorly_named_input, input_values={'x': 5, 'context_': 6}).output_value() == 11