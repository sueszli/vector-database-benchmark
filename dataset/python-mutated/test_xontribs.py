"""xontrib tests, such as they are"""
import sys
import pytest
from xonsh.xontribs import xontrib_context, xontribs_load, xontribs_loaded, xontribs_main, xontribs_reload, xontribs_unload

@pytest.fixture
def tmpmod(tmpdir):
    if False:
        return 10
    '\n    Same as tmpdir but also adds/removes it to the front of sys.path.\n\n    Also cleans out any modules loaded as part of the test.\n    '
    sys.path.insert(0, str(tmpdir))
    loadedmods = set(sys.modules.keys())
    try:
        yield tmpdir
    finally:
        del sys.path[0]
        newmods = set(sys.modules.keys()) - loadedmods
        for m in newmods:
            del sys.modules[m]

def test_noall(tmpmod):
    if False:
        return 10
    "\n    Tests what get's exported from a module without __all__\n    "
    with tmpmod.mkdir('xontrib').join('spameggs.py').open('w') as x:
        x.write('\nspam = 1\neggs = 2\n_foobar = 3\n')
    ctx = xontrib_context('spameggs')
    assert ctx == {'spam': 1, 'eggs': 2}

def test_withall(tmpmod):
    if False:
        for i in range(10):
            print('nop')
    "\n    Tests what get's exported from a module with __all__\n    "
    with tmpmod.mkdir('xontrib').join('spameggs.py').open('w') as x:
        x.write("\n__all__ = 'spam', '_foobar'\nspam = 1\neggs = 2\n_foobar = 3\n")
    ctx = xontrib_context('spameggs')
    assert ctx == {'spam': 1, '_foobar': 3}

def test_xshxontrib(tmpmod):
    if False:
        i = 10
        return i + 15
    '\n    Test that .xsh xontribs are loadable\n    '
    with tmpmod.mkdir('xontrib').join('script.xsh').open('w') as x:
        x.write("\nhello = 'world'\n")
    ctx = xontrib_context('script')
    assert ctx == {'hello': 'world'}

def test_xontrib_load(tmpmod):
    if False:
        print('Hello World!')
    '\n    Test that .xsh xontribs are loadable\n    '
    with tmpmod.mkdir('xontrib').join('script.xsh').open('w') as x:
        x.write("\nhello = 'world'\n")
    xontribs_load(['script'])
    assert 'script' in xontribs_loaded()

def test_xontrib_unload(tmpmod, xession):
    if False:
        i = 10
        return i + 15
    with tmpmod.mkdir('xontrib').join('script.py').open('w') as x:
        x.write("\nhello = 'world'\n\ndef _unload_xontrib_(xsh): del xsh.ctx['hello']\n")
    xontribs_load(['script'])
    assert 'script' in xontribs_loaded()
    assert 'hello' in xession.ctx
    xontribs_unload(['script'])
    assert 'script' not in xontribs_loaded()
    assert 'hello' not in xession.ctx

def test_xontrib_reload(tmpmod, xession):
    if False:
        i = 10
        return i + 15
    with tmpmod.mkdir('xontrib').join('script.py').open('w') as x:
        x.write("\nhello = 'world'\n\ndef _unload_xontrib_(xsh): del xsh.ctx['hello']\n")
    xontribs_load(['script'])
    assert 'script' in xontribs_loaded()
    assert xession.ctx['hello'] == 'world'
    with tmpmod.join('xontrib').join('script.py').open('w') as x:
        x.write("\nhello = 'world1'\n\ndef _unload_xontrib_(xsh): del xsh.ctx['hello']\n")
    xontribs_reload(['script'])
    assert 'script' in xontribs_loaded()
    assert xession.ctx['hello'] == 'world1'

def test_xontrib_load_dashed(tmpmod):
    if False:
        i = 10
        return i + 15
    '\n    Test that .xsh xontribs are loadable\n    '
    with tmpmod.mkdir('xontrib').join('scri-pt.xsh').open('w') as x:
        x.write("\nhello = 'world'\n")
    xontribs_load(['scri-pt'])
    assert 'scri-pt' in xontribs_loaded()

def test_xontrib_list(xession, capsys):
    if False:
        for i in range(10):
            print('nop')
    xontribs_main(['list'])
    (out, err) = capsys.readouterr()
    assert 'coreutils' in out