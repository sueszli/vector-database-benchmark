from dagster import file_relative_path, graph
from dagster_shell import create_shell_script_op

@graph
def my_graph():
    if False:
        i = 10
        return i + 15
    a = create_shell_script_op(file_relative_path(__file__, 'hello_world.sh'), name='a')
    a()