from dagster import DagsterType, In, Out, graph, job, op
from dagster._core.storage.io_manager import IOManager, io_manager

def named_io_manager(storage_dict, name):
    if False:
        i = 10
        return i + 15

    @io_manager
    def my_io_manager(_):
        if False:
            return 10

        class MyIOManager(IOManager):

            def handle_output(self, context, obj):
                if False:
                    return 10
                storage_dict[tuple(context.get_run_scoped_output_identifier())] = {'value': obj, 'output_manager_name': name}

            def load_input(self, context):
                if False:
                    while True:
                        i = 10
                result = storage_dict[tuple(context.upstream_output.get_run_scoped_output_identifier())]
                return {**result, 'input_manager_name': name}
        return MyIOManager()
    return my_io_manager

def test_graph_output():
    if False:
        for i in range(10):
            print('nop')

    @op(out=Out(io_manager_key='inner_manager'))
    def my_op(_):
        if False:
            for i in range(10):
                print('nop')
        return 5

    @op(ins={'x': In()}, out=Out(io_manager_key='inner_manager'))
    def my_op_takes_input(_, x):
        if False:
            while True:
                i = 10
        return x
    storage_dict = {}

    @graph
    def my_graph():
        if False:
            while True:
                i = 10
        return my_op_takes_input(my_op())

    @job(resource_defs={'inner_manager': named_io_manager(storage_dict, 'inner')})
    def my_job():
        if False:
            for i in range(10):
                print('nop')
        my_graph()
    result = my_job.execute_in_process()
    assert result.success
    assert storage_dict[result.run_id, 'my_graph.my_op_takes_input', 'result']['value'] == {'value': 5, 'output_manager_name': 'inner', 'input_manager_name': 'inner'}

def test_graph_upstream_output():
    if False:
        while True:
            i = 10

    @op(out=Out(io_manager_key='inner_manager'))
    def my_op(_):
        if False:
            return 10
        return 5

    @graph
    def my_graph():
        if False:
            return 10
        return my_op()

    @op
    def downstream_op(_, input1):
        if False:
            return 10
        assert input1 == {'value': 5, 'output_manager_name': 'inner', 'input_manager_name': 'inner'}
    storage_dict = {}

    @job(resource_defs={'inner_manager': named_io_manager(storage_dict, 'inner')})
    def my_job():
        if False:
            print('Hello World!')
        downstream_op(my_graph())
    result = my_job.execute_in_process()
    assert result.success

def test_io_manager_config_inside_composite():
    if False:
        while True:
            i = 10
    stored_dict = {}

    @io_manager(output_config_schema={'output_suffix': str})
    def inner_manager(_):
        if False:
            while True:
                i = 10

        class MyHardcodedIOManager(IOManager):

            def handle_output(self, context, obj):
                if False:
                    i = 10
                    return i + 15
                keys = tuple(context.get_run_scoped_output_identifier() + [context.config['output_suffix']])
                stored_dict[keys] = obj

            def load_input(self, context):
                if False:
                    print('Hello World!')
                keys = tuple(context.upstream_output.get_run_scoped_output_identifier() + [context.upstream_output.config['output_suffix']])
                return stored_dict[keys]
        return MyHardcodedIOManager()

    @op(out=Out(io_manager_key='inner_manager'))
    def my_op(_):
        if False:
            return 10
        return 'hello'

    @op
    def my_op_takes_input(_, x):
        if False:
            while True:
                i = 10
        assert x == 'hello'
        return x

    @graph
    def my_graph():
        if False:
            print('Hello World!')
        return my_op_takes_input(my_op())

    @job(resource_defs={'inner_manager': inner_manager})
    def my_job():
        if False:
            for i in range(10):
                print('nop')
        my_graph()
    result = my_job.execute_in_process(run_config={'ops': {'my_graph': {'ops': {'my_op': {'outputs': {'result': {'output_suffix': 'my_suffix'}}}}}}})
    assert result.success
    assert result.output_for_node('my_graph.my_op') == 'hello'
    assert stored_dict.get((result.run_id, 'my_graph.my_op', 'result', 'my_suffix')) == 'hello'

def test_inner_inputs_connected_to_outer_dependency():
    if False:
        for i in range(10):
            print('nop')
    my_dagster_type = DagsterType(name='foo', type_check_fn=lambda _, _a: True)

    @op(ins={'data': In(my_dagster_type)})
    def inner_op(data):
        if False:
            for i in range(10):
                print('nop')
        return data

    @graph
    def my_graph(data):
        if False:
            for i in range(10):
                print('nop')
        return inner_op(data)

    @op
    def top_level_op():
        if False:
            i = 10
            return i + 15
        return 'from top_level_op'

    @job
    def my_job():
        if False:
            i = 10
            return i + 15
        my_graph(top_level_op())
    result = my_job.execute_in_process()
    assert result.success
    assert result.output_for_node('my_graph.inner_op') == 'from top_level_op'

def test_inner_inputs_connected_to_nested_outer_dependency():
    if False:
        i = 10
        return i + 15
    my_dagster_type = DagsterType(name='foo', type_check_fn=lambda _, _a: True)

    @op(ins={'data': In(my_dagster_type)})
    def inner_op(data):
        if False:
            for i in range(10):
                print('nop')
        return data

    @graph
    def inner_graph(data_1):
        if False:
            for i in range(10):
                print('nop')
        return inner_op(data_1)

    @graph
    def middle_graph(data_2):
        if False:
            for i in range(10):
                print('nop')
        return inner_graph(data_2)

    @graph
    def outer_graph(data_3):
        if False:
            i = 10
            return i + 15
        return middle_graph(data_3)

    @op
    def top_level_op():
        if False:
            i = 10
            return i + 15
        return 'from top_level_op'

    @job
    def my_job():
        if False:
            return 10
        outer_graph(top_level_op())
    result = my_job.execute_in_process()
    assert result.success
    assert result.output_for_node('outer_graph.middle_graph.inner_graph.inner_op') == 'from top_level_op'