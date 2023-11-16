import pytest
from dagster import job, op, validate_run_config
from dagster._core.errors import DagsterInvalidConfigError


def test_validate_run_config():
    @op
    def basic():
        pass

    @job
    def basic_job():
        basic()

    validate_run_config(basic_job)

    @op(config_schema={"foo": str})
    def requires_config(_):
        pass

    @job
    def job_requires_config():
        requires_config()

    result = validate_run_config(
        job_requires_config,
        {"ops": {"requires_config": {"config": {"foo": "bar"}}}},
    )

    assert result == {
        "ops": {"requires_config": {"config": {"foo": "bar"}, "inputs": {}, "outputs": None}},
        "execution": {
            "multi_or_in_process_executor": {
                "multiprocess": {"max_concurrent": None, "retries": {"enabled": {}}}
            }
        },
        "resources": {"io_manager": {"config": None}},
        "loggers": {},
    }

    with pytest.raises(DagsterInvalidConfigError):
        validate_run_config(job_requires_config)
