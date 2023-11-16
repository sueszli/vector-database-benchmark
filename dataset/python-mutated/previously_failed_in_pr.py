import json
from pathlib import Path
from typing import Any, Dict, List, Set
from warnings import warn
from tools.testing.target_determination.heuristics.interface import HeuristicInterface, TestPrioritizations
from tools.testing.target_determination.heuristics.utils import python_test_file_to_test_name

class PreviouslyFailedInPR(HeuristicInterface):

    def __init__(self, **kwargs: Dict[str, Any]):
        if False:
            return 10
        super().__init__(**kwargs)

    def get_test_priorities(self, tests: List[str]) -> TestPrioritizations:
        if False:
            print('Hello World!')
        critical_tests = sorted(_get_previously_failing_tests())
        test_rankings = TestPrioritizations(tests_being_ranked=tests, high_relevance=critical_tests)
        return test_rankings

    def get_prediction_confidence(self, tests: List[str]) -> Dict[str, float]:
        if False:
            while True:
                i = 10
        critical_tests = _get_previously_failing_tests()
        return {test: 1 for test in critical_tests if test in tests}

def _get_previously_failing_tests() -> Set[str]:
    if False:
        return 10
    PYTEST_FAILED_TESTS_CACHE_FILE_PATH = Path('.pytest_cache/v/cache/lastfailed')
    if not PYTEST_FAILED_TESTS_CACHE_FILE_PATH.exists():
        warn(f'No pytorch cache found at {PYTEST_FAILED_TESTS_CACHE_FILE_PATH.absolute()}')
        return set()
    with open(PYTEST_FAILED_TESTS_CACHE_FILE_PATH) as f:
        last_failed_tests = json.load(f)
    prioritized_tests = _parse_prev_failing_test_files(last_failed_tests)
    return python_test_file_to_test_name(prioritized_tests)

def _parse_prev_failing_test_files(last_failed_tests: Dict[str, bool]) -> Set[str]:
    if False:
        i = 10
        return i + 15
    prioritized_tests = set()
    for test in last_failed_tests:
        parts = test.split('::')
        if len(parts) > 1:
            test_file = parts[0]
            prioritized_tests.add(test_file)
    return prioritized_tests