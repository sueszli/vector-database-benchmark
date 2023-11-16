import yaml
from dagster import graph, op
from dagster._utils import file_relative_path
from docs_snippets.concepts.configuration.config_map_example import unsigned_s3_session
from docs_snippets.concepts.configuration.configured_example import east_unsigned_s3_session, s3_session, west_signed_s3_session, west_unsigned_s3_session

def test_config_map_example():
    if False:
        for i in range(10):
            print('nop')
    execute_job_with_resource_def(unsigned_s3_session, run_config={'resources': {'key': {'config': {'region': 'us-east-1'}}}})

def execute_job_with_resource_def(resource_def, run_config=None):
    if False:
        print('Hello World!')

    @op(required_resource_keys={'key'})
    def a_op():
        if False:
            while True:
                i = 10
        pass

    @graph
    def a_graph():
        if False:
            i = 10
            return i + 15
        a_op()
    res = a_graph.to_job(resource_defs={'key': resource_def}, config=run_config).execute_in_process()
    assert res.success

def test_configured_example():
    if False:
        print('Hello World!')
    execute_job_with_resource_def(east_unsigned_s3_session)
    execute_job_with_resource_def(west_unsigned_s3_session)
    execute_job_with_resource_def(west_signed_s3_session)

def test_configured_example_yaml():
    if False:
        return 10
    with open(file_relative_path(__file__, '../../../docs_snippets/concepts/configuration/configured_example.yaml'), 'r', encoding='utf8') as fd:
        run_config = yaml.safe_load(fd.read())
    execute_job_with_resource_def(s3_session, run_config)