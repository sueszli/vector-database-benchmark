import dagstermill
dagstermill.yield_result(3, output_name='my_output')
from dagstermill import ConfigurableLocalOutputNotebookIOManager, define_dagstermill_op
from dagster import Out, file_relative_path, job, op
my_notebook_op = define_dagstermill_op(name='my_notebook', notebook_path=file_relative_path(__file__, './notebooks/my_notebook.ipynb'), output_notebook_name='output_notebook', outs={'my_output': Out(int)})

@op
def add_two(x):
    if False:
        while True:
            i = 10
    return x + 2

@job(resource_defs={'output_notebook_io_manager': ConfigurableLocalOutputNotebookIOManager()})
def my_job():
    if False:
        while True:
            i = 10
    (three, _) = my_notebook_op()
    add_two(three)