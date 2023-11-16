from dagster import graph, job, op, OpExecutionContext
from .unnested_ops import add_thirty_two, log_number, multiply_by_one_point_eight, return_fifty

@graph
def celsius_to_fahrenheit(number):
    if False:
        print('Hello World!')
    return add_thirty_two(multiply_by_one_point_eight(number))

@job
def all_together_nested():
    if False:
        i = 10
        return i + 15
    log_number(celsius_to_fahrenheit(return_fifty()))

@op(config_schema={'n': float})
def add_n(context: OpExecutionContext, number):
    if False:
        i = 10
        return i + 15
    return number + context.op_config['n']

@op(config_schema={'m': float})
def multiply_by_m(context: OpExecutionContext, number):
    if False:
        for i in range(10):
            print('nop')
    return number * context.op_config['m']

@graph
def add_n_times_m_graph(number):
    if False:
        return 10
    return multiply_by_m(add_n(number))

@job
def subgraph_config_job():
    if False:
        i = 10
        return i + 15
    add_n_times_m_graph(return_fifty())
from dagster import GraphOut

@op
def echo(i):
    if False:
        while True:
            i = 10
    print(i)

@op
def one() -> int:
    if False:
        i = 10
        return i + 15
    return 1

@op
def hello() -> str:
    if False:
        return 10
    return 'hello'

@graph(out={'x': GraphOut(), 'y': GraphOut()})
def graph_with_multiple_outputs():
    if False:
        for i in range(10):
            print('nop')
    x = one()
    y = hello()
    return {'x': x, 'y': y}

@job
def subgraph_multiple_outputs_job():
    if False:
        for i in range(10):
            print('nop')
    (x, y) = graph_with_multiple_outputs()
    echo(x)
    echo(y)