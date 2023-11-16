import pytest
from dagster import Failure, graph
from docs_snippets.concepts.ops_jobs_graphs.op_events import my_asset_op, my_expectation_op, my_failure_op, my_metadata_expectation_op, my_metadata_output, my_multiple_generic_output_op, my_op_yields, my_output_generic_op, my_output_op, my_retry_op

def execute_op_in_graph(an_op, **kwargs):
    if False:
        while True:
            i = 10

    @graph
    def my_graph():
        if False:
            return 10
        if kwargs:
            return an_op(**kwargs)
        else:
            return an_op()
    result = my_graph.execute_in_process()
    return result

def generate_stub_input_values(op):
    if False:
        for i in range(10):
            print('nop')
    input_values = {}
    default_values = {'String': 'abc', 'Int': 1, 'Any': []}
    input_defs = op.input_defs
    for input_def in input_defs:
        input_values[input_def.name] = default_values[str(input_def.dagster_type.display_name)]
    return input_values

def test_ops_compile_and_execute():
    if False:
        for i in range(10):
            print('nop')
    ops = [my_metadata_output, my_metadata_expectation_op, my_retry_op, my_asset_op, my_output_generic_op, my_expectation_op, my_multiple_generic_output_op, my_output_op, my_op_yields]
    for op in ops:
        input_values = generate_stub_input_values(op)
        result = execute_op_in_graph(op, **input_values)
        assert result
        assert result.success

def test_failure_op():
    if False:
        while True:
            i = 10
    with pytest.raises(Failure):
        execute_op_in_graph(my_failure_op)