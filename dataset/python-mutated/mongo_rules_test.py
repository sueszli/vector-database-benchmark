from __future__ import annotations
import pytest
from pants.engine.internals.scheduler import ExecutionError
from pants.testutil.rule_runner import QueryRule, RuleRunner
from .data_fixtures import platform, platform_samples
from .exceptions import ServiceMissingError
from .mongo_rules import MongoIsRunning, UsesMongoRequest, rules as mongo_rules
from .platform_rules import Platform

@pytest.fixture
def rule_runner() -> RuleRunner:
    if False:
        i = 10
        return i + 15
    return RuleRunner(rules=[*mongo_rules(), QueryRule(MongoIsRunning, (UsesMongoRequest, Platform))], target_types=[])

def run_mongo_is_running(rule_runner: RuleRunner, uses_mongo_request: UsesMongoRequest, mock_platform: Platform, *, extra_args: list[str] | None=None) -> MongoIsRunning:
    if False:
        return 10
    rule_runner.set_options(['--backend-packages=uses_services', *(extra_args or ())], env_inherit={'PATH', 'PYENV_ROOT', 'HOME'})
    result = rule_runner.request(MongoIsRunning, [uses_mongo_request, mock_platform])
    return result

def test_mongo_is_running(rule_runner: RuleRunner) -> None:
    if False:
        while True:
            i = 10
    request = UsesMongoRequest()
    mock_platform = platform(os='TestMock')
    is_running = run_mongo_is_running(rule_runner, request, mock_platform)
    assert is_running

@pytest.mark.parametrize('mock_platform', platform_samples)
def test_mongo_not_running(rule_runner: RuleRunner, mock_platform: Platform) -> None:
    if False:
        for i in range(10):
            print('nop')
    request = UsesMongoRequest(db_host='127.100.20.7', db_port=10, db_connection_timeout=10)
    with pytest.raises(ExecutionError) as exception_info:
        run_mongo_is_running(rule_runner, request, mock_platform)
    execution_error = exception_info.value
    assert len(execution_error.wrapped_exceptions) == 1
    exc = execution_error.wrapped_exceptions[0]
    assert isinstance(exc, ServiceMissingError)
    assert exc.service == 'mongo'
    assert 'The mongo service does not seem to be running' in str(exc)
    assert exc.instructions != ''