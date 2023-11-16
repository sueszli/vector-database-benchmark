import pytest
from sentry.utils.glob import glob_match

class GlobInput:

    def __init__(self, value, pat, **kwargs):
        if False:
            i = 10
            return i + 15
        self.value = value
        self.pat = pat
        self.kwargs = kwargs

    def __call__(self):
        if False:
            for i in range(10):
                print('nop')
        return glob_match(self.value, self.pat, **self.kwargs)

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return f'<GlobInput {self.__dict__!r}>'

@pytest.mark.parametrize('glob_input,expect', [[GlobInput('hello.py', '*.py'), True], [GlobInput('hello.py', '*.js'), False], [GlobInput(None, '*.js'), False], [GlobInput(None, '*'), True], [GlobInput('foo/hello.py', '*.py'), True], [GlobInput('foo/hello.py', '*.py', doublestar=True), False], [GlobInput('foo/hello.py', '**/*.py', doublestar=True), True], [GlobInput('foo/hello.PY', '**/*.py'), False], [GlobInput('foo/hello.PY', '**/*.py', doublestar=True), False], [GlobInput('foo/hello.PY', '**/*.py', ignorecase=True), True], [GlobInput('foo/hello.PY', '**/*.py', doublestar=True, ignorecase=True), True], [GlobInput('root\\foo\\hello.PY', 'root/**/*.py', ignorecase=True), False], [GlobInput('root\\foo\\hello.PY', 'root/**/*.py', doublestar=True, ignorecase=True), False], [GlobInput('root\\foo\\hello.PY', 'root/**/*.py', ignorecase=True, path_normalize=True), True], [GlobInput('root\\foo\\hello.PY', 'root/**/*.py', doublestar=True, ignorecase=True, path_normalize=True), True], [GlobInput('foo:\nbar', 'foo:*'), True], [GlobInput('foo:\nbar', 'foo:*', allow_newline=False), False]])
def test_glob_match(glob_input, expect):
    if False:
        print('Hello World!')
    assert glob_input() == expect