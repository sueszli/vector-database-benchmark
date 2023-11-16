import yaml
from dagster import job
from dagster._utils import file_relative_path
from docs_snippets.concepts.ops_jobs_graphs.graph_provides_config import celsius_to_fahrenheit
from docs_snippets.concepts.ops_jobs_graphs.graph_provides_config_mapping import to_fahrenheit
from docs_snippets.concepts.ops_jobs_graphs.nested_graphs import all_together_nested, subgraph_config_job, subgraph_multiple_outputs_job
from docs_snippets.concepts.ops_jobs_graphs.unnested_ops import all_together_unnested, return_fifty

def test_unnested():
    if False:
        return 10
    assert all_together_unnested.execute_in_process().success

def test_nested():
    if False:
        return 10
    assert all_together_nested.execute_in_process().success

def test_composite_config():
    if False:
        print('Hello World!')
    with open(file_relative_path(__file__, '../../../docs_snippets/concepts/ops_jobs_graphs/composite_config.yaml'), 'r', encoding='utf8') as fd:
        run_config = yaml.safe_load(fd.read())
    assert subgraph_config_job.execute_in_process(run_config=run_config).success

def test_graph_provides_config():
    if False:
        return 10

    @job
    def my_job():
        if False:
            print('Hello World!')
        celsius_to_fahrenheit(return_fifty())
    my_job.execute_in_process()

def test_config_mapping():
    if False:
        for i in range(10):
            print('nop')

    @job
    def my_job():
        if False:
            i = 10
            return i + 15
        to_fahrenheit(return_fifty())
    with open(file_relative_path(__file__, '../../../docs_snippets/concepts/ops_jobs_graphs/composite_config_mapping.yaml'), 'r', encoding='utf8') as fd:
        run_config = yaml.safe_load(fd.read())
    assert my_job.execute_in_process(run_config=run_config).success

def test_composite_multi_outputs():
    if False:
        print('Hello World!')
    assert subgraph_multiple_outputs_job.execute_in_process().success