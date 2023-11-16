from dagster import job, materialize
from docs_snippets.concepts.configuration.configurable_op_asset_resource import asset_using_config, op_using_config

def execute_with_config():
    if False:
        i = 10
        return i + 15

    @job
    def example_job():
        if False:
            print('Hello World!')
        op_using_config()
    job_result = example_job.execute_in_process(run_config={'ops': {'op_using_config': {'config': {'person_name': 'Alice'}}}})
    asset_result = materialize([asset_using_config], run_config={'ops': {'asset_using_config': {'config': {'person_name': 'Alice'}}}})
    assert job_result.success
    assert asset_result.success

def execute_with_bad_config():
    if False:
        for i in range(10):
            print('nop')

    @job
    def example_job():
        if False:
            for i in range(10):
                print('nop')
        op_using_config()
    op_result = example_job.execute_in_process(run_config={'ops': {'op_using_config': {'config': {'nonexistent_config_value': 1}}}})
    asset_result = materialize([asset_using_config], run_config={'ops': {'asset_using_config': {'config': {'nonexistent_config_value': 1}}}})
    assert op_result.success is False
    assert asset_result.success is False