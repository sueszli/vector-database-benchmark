import pytest
from dagster import job, op, validate_run_config
from dagster._core.errors import DagsterInvalidConfigError

def test_validate_run_config():
    if False:
        while True:
            i = 10

    @op
    def basic():
        if False:
            return 10
        pass

    @job
    def basic_job():
        if False:
            return 10
        basic()
    validate_run_config(basic_job)

    @op(config_schema={'foo': str})
    def requires_config(_):
        if False:
            i = 10
            return i + 15
        pass

    @job
    def job_requires_config():
        if False:
            while True:
                i = 10
        requires_config()
    result = validate_run_config(job_requires_config, {'ops': {'requires_config': {'config': {'foo': 'bar'}}}})
    assert result == {'ops': {'requires_config': {'config': {'foo': 'bar'}, 'inputs': {}, 'outputs': None}}, 'execution': {'multi_or_in_process_executor': {'multiprocess': {'max_concurrent': None, 'retries': {'enabled': {}}}}}, 'resources': {'io_manager': {'config': None}}, 'loggers': {}}
    with pytest.raises(DagsterInvalidConfigError):
        validate_run_config(job_requires_config)