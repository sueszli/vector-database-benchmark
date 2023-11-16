import pytest
from dagster import DagsterInvalidConfigError
from docs_snippets.concepts.configuration.execute_with_config import execute_with_bad_config, execute_with_config

def test_run_good_example():
    if False:
        while True:
            i = 10
    execute_with_config()

def test_run_bad_example():
    if False:
        while True:
            i = 10
    with pytest.raises(DagsterInvalidConfigError):
        execute_with_bad_config()