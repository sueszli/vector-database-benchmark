import os
from pathlib import Path
from typing import Iterator, List, Tuple
import pytest
from pip._vendor.resolvelib import BaseReporter, Resolver
from pip._internal.resolution.resolvelib.base import Candidate, Constraint, Requirement
from pip._internal.resolution.resolvelib.factory import Factory
from pip._internal.resolution.resolvelib.provider import PipProvider
from tests.lib import TestData

@pytest.fixture
def test_cases(data: TestData) -> Iterator[List[Tuple[str, str, int]]]:
    if False:
        print('Hello World!')

    def _data_file(name: str) -> Path:
        if False:
            print('Hello World!')
        return data.packages.joinpath(name)

    def data_file(name: str) -> str:
        if False:
            print('Hello World!')
        return os.fspath(_data_file(name))

    def data_url(name: str) -> str:
        if False:
            while True:
                i = 10
        return _data_file(name).as_uri()
    test_cases = [('simple', 'simple', 3), ('simple>1.0', 'simple', 2), (data_file('simplewheel-1.0-py2.py3-none-any.whl'), 'simplewheel', 1), (data_url('simplewheel-1.0-py2.py3-none-any.whl'), 'simplewheel', 1), (data_file('simple-1.0.tar.gz'), 'simple', 1), (data_url('simple-1.0.tar.gz'), 'simple', 1)]
    yield test_cases

def test_new_resolver_requirement_has_name(test_cases: List[Tuple[str, str, int]], factory: Factory) -> None:
    if False:
        print('Hello World!')
    'All requirements should have a name'
    for (spec, name, _) in test_cases:
        reqs = list(factory.make_requirements_from_spec(spec, comes_from=None))
        assert len(reqs) == 1
        assert reqs[0].name == name

def test_new_resolver_correct_number_of_matches(test_cases: List[Tuple[str, str, int]], factory: Factory) -> None:
    if False:
        while True:
            i = 10
    'Requirements should return the correct number of candidates'
    for (spec, _, match_count) in test_cases:
        reqs = list(factory.make_requirements_from_spec(spec, comes_from=None))
        assert len(reqs) == 1
        req = reqs[0]
        matches = factory.find_candidates(req.name, {req.name: [req]}, {}, Constraint.empty(), prefers_installed=False)
        assert sum((1 for _ in matches)) == match_count

def test_new_resolver_candidates_match_requirement(test_cases: List[Tuple[str, str, int]], factory: Factory) -> None:
    if False:
        print('Hello World!')
    'Candidates returned from find_candidates should satisfy the requirement'
    for (spec, _, _) in test_cases:
        reqs = list(factory.make_requirements_from_spec(spec, comes_from=None))
        assert len(reqs) == 1
        req = reqs[0]
        candidates = factory.find_candidates(req.name, {req.name: [req]}, {}, Constraint.empty(), prefers_installed=False)
        for c in candidates:
            assert isinstance(c, Candidate)
            assert req.is_satisfied_by(c)

def test_new_resolver_full_resolve(factory: Factory, provider: PipProvider) -> None:
    if False:
        return 10
    'A very basic full resolve'
    reqs = list(factory.make_requirements_from_spec('simplewheel', comes_from=None))
    assert len(reqs) == 1
    r: Resolver[Requirement, Candidate, str] = Resolver(provider, BaseReporter())
    result = r.resolve(reqs)
    assert set(result.mapping.keys()) == {'simplewheel'}