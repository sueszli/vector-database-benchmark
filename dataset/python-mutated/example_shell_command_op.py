from dagster import graph
from dagster_shell import create_shell_command_op

@graph
def my_graph():
    if False:
        i = 10
        return i + 15
    a = create_shell_command_op('echo "hello, world!"', name='a')
    a()