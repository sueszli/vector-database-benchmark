from __future__ import annotations
import dataclasses
import pytest
from pants.testutil.rule_runner import QueryRule, RuleRunner
from .platform_rules import Platform, rules as platform_rules

@pytest.fixture
def rule_runner() -> RuleRunner:
    if False:
        print('Hello World!')
    return RuleRunner(rules=[*platform_rules(), QueryRule(Platform, ())], target_types=[])

def test_get_platform(rule_runner: RuleRunner) -> None:
    if False:
        print('Hello World!')
    rule_runner.set_options(['--backend-packages=uses_services'], env_inherit={'PATH', 'PYENV_ROOT', 'HOME'})
    platform = rule_runner.request(Platform, ())
    assert isinstance(platform, Platform)
    assert dataclasses.is_dataclass(platform)