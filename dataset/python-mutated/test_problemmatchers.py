import re
import pytest
from scripts.dev.ci import problemmatchers

@pytest.mark.parametrize('matcher_name', list(problemmatchers.MATCHERS))
def test_patterns(matcher_name):
    if False:
        while True:
            i = 10
    "Make sure all regexps are valid.\n\n    They aren't actually Python syntax, but hopefully close enough to it to compile with\n    Python's re anyways.\n    "
    for matcher in problemmatchers.MATCHERS[matcher_name]:
        for pattern in matcher['pattern']:
            regexp = pattern['regexp']
            print(regexp)
            re.compile(regexp)