import pytest
from dagster import Any, DagsterInvalidDefinitionError, DependencyDefinition, GraphDefinition, In, InputMapping, Int, List, MultiDependencyDefinition, Nothing, Out, OutputMapping, graph, job, op
from dagster._core.definitions.composition import MappedInputPlaceholder

def test_simple_values():
    if False:
        i = 10
        return i + 15

    @op(ins={'numbers': In(List[Int])})
    def sum_num(_context, numbers):
        if False:
            while True:
                i = 10
        assert set(numbers) == set([1, 2, 3])
        return sum(numbers)

    @op
    def emit_1():
        if False:
            while True:
                i = 10
        return 1

    @op
    def emit_2():
        if False:
            print('Hello World!')
        return 2

    @op
    def emit_3():
        if False:
            return 10
        return 3
    foo_job = GraphDefinition(name='input_test', node_defs=[emit_1, emit_2, emit_3, sum_num], dependencies={'sum_num': {'numbers': MultiDependencyDefinition([DependencyDefinition('emit_1'), DependencyDefinition('emit_2'), DependencyDefinition('emit_3')])}}).to_job()
    result = foo_job.execute_in_process()
    assert result.success
    assert result.output_for_node('sum_num') == 6

@op(ins={'stuff': In(List[Any])})
def collect(_context, stuff):
    if False:
        print('Hello World!')
    assert set(stuff) == set([1, None, 'one'])
    return stuff

@op
def emit_num():
    if False:
        print('Hello World!')
    return 1

@op
def emit_none():
    if False:
        for i in range(10):
            print('nop')
    pass

@op
def emit_str():
    if False:
        return 10
    return 'one'

@op(out=Out(Nothing))
def emit_nothing():
    if False:
        i = 10
        return i + 15
    pass

def test_interleaved_values():
    if False:
        i = 10
        return i + 15
    foo_job = GraphDefinition(name='input_test', node_defs=[emit_num, emit_none, emit_str, collect], dependencies={'collect': {'stuff': MultiDependencyDefinition([DependencyDefinition('emit_num'), DependencyDefinition('emit_none'), DependencyDefinition('emit_str')])}}).to_job()
    result = foo_job.execute_in_process()
    assert result.success

def test_dsl():
    if False:
        print('Hello World!')

    @job
    def input_test():
        if False:
            for i in range(10):
                print('nop')
        collect([emit_num(), emit_none(), emit_str()])
    result = input_test.execute_in_process()
    assert result.success

def test_collect_one():
    if False:
        i = 10
        return i + 15

    @op
    def collect_one(list_arg):
        if False:
            return 10
        assert list_arg == ['one']

    @job
    def multi_one():
        if False:
            while True:
                i = 10
        collect_one([emit_str()])
    assert multi_one.execute_in_process().success

def test_fan_in_manual():
    if False:
        while True:
            i = 10

    @graph
    def _target_graph_dsl(str_in, none_in):
        if False:
            print('Hello World!')
        num = emit_num()
        return collect([num, str_in, none_in])
    _target_graph_manual = GraphDefinition(name='manual_graph', node_defs=[emit_num, collect], input_mappings=[InputMapping(graph_input_name='str_in', mapped_node_name='collect', mapped_node_input_name='stuff', fan_in_index=1), InputMapping(graph_input_name='none_in', mapped_node_name='collect', mapped_node_input_name='stuff', fan_in_index=2)], output_mappings=[OutputMapping(graph_output_name='result', mapped_node_name='collect', mapped_node_output_name='result')], dependencies={'collect': {'stuff': MultiDependencyDefinition([DependencyDefinition('emit_num'), MappedInputPlaceholder, MappedInputPlaceholder])}})
    with pytest.raises(DagsterInvalidDefinitionError, match='index 2 in the MultiDependencyDefinition is not a MappedInputPlaceholder'):
        _missing_placeholder = GraphDefinition(name='manual_graph', node_defs=[emit_num, collect], input_mappings=[InputMapping(graph_input_name='str_in', mapped_node_name='collect', mapped_node_input_name='stuff', fan_in_index=1), InputMapping(graph_input_name='none_in', mapped_node_name='collect', mapped_node_input_name='stuff', fan_in_index=2)], output_mappings=[OutputMapping(graph_output_name='result', mapped_node_name='collect', mapped_node_output_name='result')], dependencies={'collect': {'stuff': MultiDependencyDefinition([DependencyDefinition('emit_num'), MappedInputPlaceholder])}})
    with pytest.raises(DagsterInvalidDefinitionError, match='is not a MultiDependencyDefinition'):
        _bad_target = GraphDefinition(name='manual_graph', node_defs=[emit_num, collect], input_mappings=[InputMapping(graph_input_name='str_in', mapped_node_name='collect', mapped_node_input_name='stuff', fan_in_index=1), InputMapping(graph_input_name='none_in', mapped_node_name='collect', mapped_node_input_name='stuff', fan_in_index=2)], output_mappings=[OutputMapping(graph_output_name='result', mapped_node_name='collect', mapped_node_output_name='result')], dependencies={'collect': {'stuff': DependencyDefinition('emit_num')}})
    with pytest.raises(DagsterInvalidDefinitionError, match='Unsatisfied MappedInputPlaceholder at index 3'):
        _missing_placeholder = GraphDefinition(name='manual_graph', node_defs=[emit_num, collect], input_mappings=[InputMapping(graph_input_name='str_in', mapped_node_name='collect', mapped_node_input_name='stuff', fan_in_index=1), InputMapping(graph_input_name='none_in', mapped_node_name='collect', mapped_node_input_name='stuff', fan_in_index=2)], output_mappings=[OutputMapping(graph_output_name='result', mapped_node_name='collect', mapped_node_output_name='result')], dependencies={'collect': {'stuff': MultiDependencyDefinition([DependencyDefinition('emit_num'), MappedInputPlaceholder, MappedInputPlaceholder, MappedInputPlaceholder])}})

def test_nothing_deps():
    if False:
        return 10
    GraphDefinition(name='input_test', node_defs=[emit_num, emit_nothing, emit_str, collect], dependencies={'collect': {'stuff': MultiDependencyDefinition([DependencyDefinition('emit_num'), DependencyDefinition('emit_nothing'), DependencyDefinition('emit_str')])}}).to_job()