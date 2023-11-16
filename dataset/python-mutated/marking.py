"""
Custom pytest mark typings
"""
from typing import TYPE_CHECKING, Callable, List, Optional
import pytest

class AwsCompatibilityMarkers:
    validated = pytest.mark.aws_validated
    manual_setup_required = pytest.mark.aws_manual_setup_required
    needs_fixing = pytest.mark.aws_needs_fixing
    only_localstack = pytest.mark.aws_only_localstack
    unknown = pytest.mark.aws_unknown

class ParityMarkers:
    aws_validated = pytest.mark.aws_validated
    only_localstack = pytest.mark.only_localstack

class SkipSnapshotVerifyMarker:

    def __call__(self, *, paths: 'Optional[List[str]]'=None, condition: 'Optional[Callable[[...], bool]]'=None):
        if False:
            for i in range(10):
                print('nop')
        ...

class MultiRuntimeMarker:

    def __call__(self, *, scenario: str, runtimes: Optional[List[str]]=None):
        if False:
            while True:
                i = 10
        ...

class SnapshotMarkers:
    skip_snapshot_verify: SkipSnapshotVerifyMarker = pytest.mark.skip_snapshot_verify

class Markers:
    aws = AwsCompatibilityMarkers
    parity = ParityMarkers
    snapshot = SnapshotMarkers
    multiruntime: MultiRuntimeMarker = pytest.mark.multiruntime
    acceptance_test_beta = pytest.mark.acceptance_test
    skip_offline = pytest.mark.skip_offline
    only_on_amd64 = pytest.mark.only_on_amd64
    resource_heavy = pytest.mark.resource_heavy
    only_in_docker = pytest.mark.only_in_docker
if TYPE_CHECKING:
    from _pytest.config import Config

@pytest.hookimpl
def pytest_collection_modifyitems(session: pytest.Session, config: 'Config', items: List[pytest.Item]) -> None:
    if False:
        i = 10
        return i + 15
    'Enforce that each test has exactly one aws compatibility marker'
    marker_errors = []
    for item in items:
        if 'tests/aws' not in item.fspath.dirname:
            continue
        aws_markers = list()
        for mark in item.iter_markers():
            if mark.name.startswith('aws_'):
                aws_markers.append(mark.name)
        if len(aws_markers) > 1:
            marker_errors.append(f'{item.nodeid}: Too many aws markers specified: {aws_markers}')
        elif len(aws_markers) == 0:
            marker_errors.append(f'{item.nodeid}: Missing aws marker. Specify at least one marker, e.g. @markers.aws.validated')
    if marker_errors:
        raise pytest.UsageError(*marker_errors)