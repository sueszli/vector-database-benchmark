import pytest
from PyInstaller.lib.modulegraph import modulegraph
from PyInstaller import configure
from PyInstaller import __main__ as pyi_main
from PyInstaller.compat import is_win

@pytest.fixture
def large_import_chain(tmpdir):
    if False:
        return 10
    pkg = tmpdir.join('pkg')
    pkg.join('__init__.py').ensure().write('from . import a')
    mod = None
    for alpha in 'abcdefg':
        if mod:
            mod.write('import pkg.%s' % alpha)
        subpkg = pkg.join(alpha).mkdir()
        subpkg.join('__init__.py').write('from . import %s000' % alpha)
        for num in range(250):
            mod = subpkg.join('%s%03i.py' % (alpha, num))
            mod.write('from . import %s%03i' % (alpha, num + 1))
    script = tmpdir.join('script.py')
    script.write('import pkg')
    return ([str(tmpdir)], str(script))

def test_recursion_too_deep(large_import_chain):
    if False:
        return 10
    '\n    modulegraph is recursive and triggers RecursionError if nesting of imported modules is too deep.\n    This can be worked around by increasing recursion limit.\n\n    With the default recursion limit (1000), the recursion error occurs at about 115 modules, with limit 2000\n    (as tested below) at about 240 modules, and with limit 5000 at about 660 modules.\n    '
    if is_win:
        pytest.xfail('Worker is known to crash on Windows.')
    (path, script) = large_import_chain
    mg = modulegraph.ModuleGraph(path)
    with pytest.raises(RecursionError):
        mg.add_script(str(script))

def test_RecursionError_prints_message(tmpdir, large_import_chain, monkeypatch):
    if False:
        i = 10
        return i + 15
    '\n    modulegraph is recursive and triggers RecursionError if nesting of imported modules is too deep.\n    Ensure an informative message is printed if RecursionError occurs.\n    '
    if is_win:
        pytest.xfail('Worker is known to crash on Windows.')
    (path, script) = large_import_chain
    default_args = ['--specpath', str(tmpdir), '--distpath', str(tmpdir.join('dist')), '--workpath', str(tmpdir.join('build')), '--path', str(tmpdir)]
    pyi_args = [script] + default_args
    PYI_CONFIG = configure.get_config()
    PYI_CONFIG['cachedir'] = str(tmpdir)
    with pytest.raises(SystemExit) as execinfo:
        pyi_main.run(pyi_args, PYI_CONFIG)
    assert 'sys.setrecursionlimit' in str(execinfo.value)