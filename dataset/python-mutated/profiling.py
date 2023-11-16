from typing import Any, Dict, List
from tools.stats.import_test_stats import ADDITIONAL_CI_FILES_FOLDER, TD_HEURISTIC_PROFILING_FILE
from tools.testing.target_determination.heuristics.interface import HeuristicInterface, TestPrioritizations
from tools.testing.target_determination.heuristics.utils import get_correlated_tests, get_ratings_for_tests, normalize_ratings

class Profiling(HeuristicInterface):

    def __init__(self, **kwargs: Any):
        if False:
            print('Hello World!')
        super().__init__(**kwargs)

    def get_test_priorities(self, tests: List[str]) -> TestPrioritizations:
        if False:
            while True:
                i = 10
        correlated_tests = get_correlated_tests(ADDITIONAL_CI_FILES_FOLDER / TD_HEURISTIC_PROFILING_FILE)
        relevant_correlated_tests = [test for test in correlated_tests if test in tests]
        test_rankings = TestPrioritizations(tests_being_ranked=tests, probable_relevance=relevant_correlated_tests)
        return test_rankings

    def get_prediction_confidence(self, tests: List[str]) -> Dict[str, float]:
        if False:
            for i in range(10):
                print('nop')
        test_ratings = get_ratings_for_tests(ADDITIONAL_CI_FILES_FOLDER / TD_HEURISTIC_PROFILING_FILE)
        test_ratings = {k: v for (k, v) in test_ratings.items() if k in tests}
        return normalize_ratings(test_ratings, 1)