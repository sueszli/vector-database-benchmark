from __future__ import annotations
import pytest
from pants.engine.addresses import Address
from pants.engine.internals.scheduler import ExecutionError
from pants.testutil.rule_runner import RuleRunner
from .target_types import PackMetadata, PackMetadataInGitSubmodule, UnmatchedGlobsError

@pytest.fixture
def rule_runner() -> RuleRunner:
    if False:
        while True:
            i = 10
    return RuleRunner(rules=[], target_types=[PackMetadata, PackMetadataInGitSubmodule])
GIT_SUBMODULE_BUILD_FILE = '\npack_metadata_in_git_submodule(\n    name="metadata",\n    sources=["./submodule_dir/pack.yaml"],\n)\n'

def test_git_submodule_sources_missing(rule_runner: RuleRunner) -> None:
    if False:
        while True:
            i = 10
    rule_runner.write_files({'packs/BUILD': GIT_SUBMODULE_BUILD_FILE})
    with pytest.raises(ExecutionError) as e:
        _ = rule_runner.get_target(Address('packs', target_name='metadata'))
    exc = e.value.wrapped_exceptions[0]
    assert isinstance(exc, UnmatchedGlobsError)
    assert 'One or more git submodules is not checked out' in str(exc)

def test_git_submodule_sources_present(rule_runner: RuleRunner) -> None:
    if False:
        for i in range(10):
            print('nop')
    rule_runner.write_files({'packs/BUILD': GIT_SUBMODULE_BUILD_FILE, 'packs/submodule_dir/pack.yaml': '---\nname: foobar\n'})
    _ = rule_runner.get_target(Address('packs', target_name='metadata'))