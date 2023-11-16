from typing import Any, Dict, List, Set
from warnings import warn
from tools.testing.target_determination.heuristics.interface import HeuristicInterface, TestPrioritizations
from tools.testing.target_determination.heuristics.utils import python_test_file_to_test_name, query_changed_files

class EditedByPR(HeuristicInterface):

    def __init__(self, **kwargs: Dict[str, Any]):
        if False:
            while True:
                i = 10
        super().__init__(**kwargs)

    def get_test_priorities(self, tests: List[str]) -> TestPrioritizations:
        if False:
            while True:
                i = 10
        critical_tests = sorted(_get_modified_tests())
        test_rankings = TestPrioritizations(tests_being_ranked=tests, high_relevance=critical_tests)
        return test_rankings

    def get_prediction_confidence(self, tests: List[str]) -> Dict[str, float]:
        if False:
            return 10
        critical_tests = _get_modified_tests()
        return {test: 1 for test in critical_tests if test in tests}

def _get_modified_tests() -> Set[str]:
    if False:
        for i in range(10):
            print('nop')
    try:
        changed_files = query_changed_files()
    except Exception as e:
        warn(f"Can't query changed test files due to {e}")
        return set()
    return python_test_file_to_test_name(set(changed_files))