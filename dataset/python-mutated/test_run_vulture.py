import sys
import textwrap
import pytest
from tests.helpers import testutils
try:
    from scripts.dev import run_vulture
except ImportError:
    if hasattr(sys, 'frozen'):
        pass
    else:
        raise
pytestmark = [pytest.mark.not_frozen]

class VultureDir:
    """Fixture similar to pytest's testdir fixture for vulture.

    Attributes:
        _tmp_path: The pytest tmp_path fixture.
    """

    def __init__(self, tmp_path):
        if False:
            print('Hello World!')
        self._tmp_path = tmp_path

    def run(self):
        if False:
            return 10
        'Run vulture over all generated files and return the output.'
        names = [p.name for p in self._tmp_path.glob('*')]
        assert names
        with testutils.change_cwd(self._tmp_path):
            return run_vulture.run(names)

    def makepyfile(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Create a python file, similar to TestDir.makepyfile.'
        for (filename, data) in kwargs.items():
            text = textwrap.dedent(data)
            (self._tmp_path / (filename + '.py')).write_text(text, 'utf-8')

@pytest.fixture
def vultdir(tmp_path):
    if False:
        print('Hello World!')
    return VultureDir(tmp_path)

def test_used(vultdir):
    if False:
        while True:
            i = 10
    vultdir.makepyfile(foo='\n        def foo():\n            pass\n\n        foo()\n    ')
    assert not vultdir.run()

def test_unused_func(vultdir):
    if False:
        while True:
            i = 10
    vultdir.makepyfile(foo='\n        def foo():\n            pass\n    ')
    msg = "*foo.py:2: unused function 'foo' (60% confidence)"
    msgs = vultdir.run()
    assert len(msgs) == 1
    assert testutils.pattern_match(pattern=msg, value=msgs[0])

def test_unused_method_camelcase(vultdir):
    if False:
        print('Hello World!')
    'Should be ignored because those are Qt methods.'
    vultdir.makepyfile(foo='\n        class Foo():\n\n            def fooBar(self):\n                pass\n\n        Foo()\n    ')
    assert not vultdir.run()