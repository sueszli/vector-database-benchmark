import re
import pytest
import pycodestyle
from testing.support import errors_from_src
SELFTEST_REGEX = re.compile('\\b(Okay|[EW]\\d{3}): (.*)')

def get_tests():
    if False:
        i = 10
        return i + 15
    ret = [pytest.param(match[1], match[2], id=f'pycodestyle.py:{f.__code__.co_firstlineno}:{f.__name__}@{i}') for group in pycodestyle._checks.values() for f in group if f.__doc__ is not None for (i, match) in enumerate(SELFTEST_REGEX.finditer(f.__doc__))]
    assert ret
    return tuple(ret)

@pytest.mark.parametrize(('expected', 's'), get_tests())
def test(expected, s):
    if False:
        print('Hello World!')
    s = '\n'.join((*s.replace('\\t', '\t').split('\\n'), ''))
    errors = errors_from_src(s)
    if expected == 'Okay':
        assert errors == []
    else:
        for error in errors:
            if error.startswith(f'{expected}:'):
                break
        else:
            raise AssertionError(f'expected {expected} from {s!r}')