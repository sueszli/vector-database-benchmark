import types
import pytest
import itertools
from textwrap import dedent
from PyInstaller import HOMEPATH
from PyInstaller.depend import analysis
from PyInstaller.lib.modulegraph import modulegraph
import PyInstaller.log as logging
from PyInstaller.utils.tests import gen_sourcefile

def test_get_co_using_ctypes(tmpdir):
    if False:
        return 10
    logging.logger.setLevel(logging.DEBUG)
    mg = analysis.PyiModuleGraph(HOMEPATH, excludes=['xencodings'])
    script = tmpdir.join('script.py')
    script.write('import ctypes')
    script_filename = str(script)
    mg.add_script(script_filename)
    res = mg.get_code_using('ctypes')
    assert script_filename in res
    assert isinstance(res[script_filename], types.CodeType), res

def test_get_co_using_ctypes_from_extension():
    if False:
        while True:
            i = 10
    logging.logger.setLevel(logging.DEBUG)
    mg = analysis.PyiModuleGraph(HOMEPATH, excludes=['xencodings'])
    struct = mg.createNode(modulegraph.Extension, '_struct', 'struct.so')
    mg.implyNodeReference(struct, 'ctypes')
    res = mg.get_code_using('ctypes')
    assert '_struct' not in res

def test_metadata_collection(tmpdir):
    if False:
        i = 10
        return i + 15
    from PyInstaller.utils.hooks import copy_metadata
    mg = analysis.PyiModuleGraph(HOMEPATH, excludes=['xencodings'])
    script = tmpdir.join('script.py')
    script.write(dedent('\n            from importlib.metadata import distribution, version\n            import importlib.metadata\n\n            distribution("setuptools")\n            importlib.metadata.version("altgraph")\n            '))
    mg.add_script(str(script))
    metadata = mg.metadata_required()
    assert copy_metadata('setuptools')[0] in metadata
    assert copy_metadata('altgraph')[0] in metadata

class FakePyiModuleGraph(analysis.PyiModuleGraph):

    def _analyze_base_modules(self):
        if False:
            while True:
                i = 10
        self._base_modules = ()

@pytest.fixture
def fresh_pyi_modgraph(monkeypatch):
    if False:
        print('Hello World!')
    '\n    Get a fresh PyiModuleGraph\n    '

    def fake_base_modules(self):
        if False:
            return 10
        self._base_modules = ()
    logging.logger.setLevel(logging.DEBUG)
    monkeypatch.setattr(analysis, '_cached_module_graph_', None)
    monkeypatch.setattr(analysis.PyiModuleGraph, '_analyze_base_modules', fake_base_modules)
    return analysis.initialize_modgraph()

def test_cached_graph_is_not_leaking(fresh_pyi_modgraph, monkeypatch, tmpdir):
    if False:
        while True:
            i = 10
    '\n    Ensure cached PyiModulegraph can separate imports between scripts.\n    '
    mg = fresh_pyi_modgraph
    src = gen_sourcefile(tmpdir, 'print', test_id='1')
    mg.add_script(str(src))
    assert not mg.find_node('uuid')
    src = gen_sourcefile(tmpdir, 'import uuid', test_id='2')
    node = mg.add_script(str(src))
    assert node is not None
    names = [n.identifier for n in mg.iter_graph(start=node)]
    assert 'uuid' in names
    src = gen_sourcefile(tmpdir, 'print', test_id='3')
    node = mg.add_script(str(src))
    assert node is not None
    names = [n.identifier for n in mg.iter_graph(start=node)]
    assert 'uuid' not in names

def test_cached_graph_is_not_leaking_hidden_imports(fresh_pyi_modgraph, tmpdir):
    if False:
        while True:
            i = 10
    '\n    Ensure cached PyiModulegraph can separate hidden imports between scripts.\n    '
    mg = fresh_pyi_modgraph
    src = gen_sourcefile(tmpdir, 'print', test_id='2')
    node = mg.add_script(str(src))
    assert node is not None
    mg.add_hiddenimports(['uuid'])
    names = [n.identifier for n in mg.iter_graph(start=node)]
    assert 'uuid' in names
    src = gen_sourcefile(tmpdir, 'print', test_id='3')
    node = mg.add_script(str(src))
    assert node is not None
    names = [n.identifier for n in mg.iter_graph(start=node)]
    assert 'uuid' not in names

def test_graph_collects_script_dependencies(fresh_pyi_modgraph, tmpdir):
    if False:
        return 10
    mg = fresh_pyi_modgraph
    src1 = gen_sourcefile(tmpdir, 'print', test_id='1')
    node = mg.add_script(str(src1))
    assert node is not None
    assert not mg.find_node('uuid')
    src2 = gen_sourcefile(tmpdir, 'import uuid', test_id='2')
    mg.add_script(str(src2))
    assert mg.find_node('uuid')
    names = [n.identifier for n in mg.iter_graph(start=node)]
    assert str(src2) in names
    assert 'uuid' in names

def _gen_pseudo_rthooks(name, rthook_dat, tmpdir, gen_files=True):
    if False:
        print('Hello World!')
    hd = tmpdir.ensure(name, dir=True)
    if gen_files:
        for fn in itertools.chain(*rthook_dat.values()):
            hd.ensure('rthooks', fn)
    rhd = hd.ensure('rthooks.dat')
    rhd.write(repr(rthook_dat))
    return hd

def test_collect_rthooks_1(tmpdir, monkeypatch):
    if False:
        i = 10
        return i + 15
    rh1 = {'test_pyimodulegraph_mymod1': ['m1.py']}
    hd1 = _gen_pseudo_rthooks('h1', rh1, tmpdir)
    mg = FakePyiModuleGraph(HOMEPATH, user_hook_dirs=[str(hd1)])
    assert len(mg._available_rthooks['test_pyimodulegraph_mymod1']) == 1

def test_collect_rthooks_2(tmpdir, monkeypatch):
    if False:
        for i in range(10):
            print('nop')
    rh1 = {'test_pyimodulegraph_mymod1': ['m1.py']}
    rh2 = {'test_pyimodulegraph_mymod2': ['rth1.py', 'rth1.py']}
    hd1 = _gen_pseudo_rthooks('h1', rh1, tmpdir)
    hd2 = _gen_pseudo_rthooks('h2', rh2, tmpdir)
    mg = FakePyiModuleGraph(HOMEPATH, user_hook_dirs=[str(hd1), str(hd2)])
    assert len(mg._available_rthooks['test_pyimodulegraph_mymod1']) == 1
    assert len(mg._available_rthooks['test_pyimodulegraph_mymod2']) == 2

def test_collect_rthooks_3(tmpdir, monkeypatch):
    if False:
        print('Hello World!')
    rh1 = {'test_pyimodulegraph_mymod1': ['m1.py']}
    rh2 = {'test_pyimodulegraph_mymod1': ['rth1.py', 'rth1.py']}
    hd1 = _gen_pseudo_rthooks('h1', rh1, tmpdir)
    hd2 = _gen_pseudo_rthooks('h2', rh2, tmpdir)
    mg = FakePyiModuleGraph(HOMEPATH, user_hook_dirs=[str(hd1), str(hd2)])
    assert len(mg._available_rthooks['test_pyimodulegraph_mymod1']) == 1

def test_collect_rthooks_fail_1(tmpdir, monkeypatch):
    if False:
        return 10
    rh1 = {'test_pyimodulegraph_mymod1': ['m1.py']}
    hd1 = _gen_pseudo_rthooks('h1', rh1, tmpdir, False)
    with pytest.raises(AssertionError):
        FakePyiModuleGraph(HOMEPATH, user_hook_dirs=[str(hd1)])

class FakeGraph(analysis.PyiModuleGraph):
    """
    A simplified module graph containing a single node module *foo* with user-defined content.
    """

    def __init__(self, source):
        if False:
            i = 10
            return i + 15
        self.code = compile(source, '<>', 'exec')

    def get_code_using(self, package):
        if False:
            while True:
                i = 10
        return {'foo': self.code}

def test_metadata_searching():
    if False:
        i = 10
        return i + 15
    '\n    Test the top level for bytecode scanning for metadata requirements.\n    '
    from PyInstaller.utils.hooks import copy_metadata
    pyinstaller = set(copy_metadata('pyinstaller'))
    with_dependencies = set(copy_metadata('pyinstaller', recursive=True))
    self = FakeGraph("from importlib.metadata import distribution; distribution('pyinstaller')")
    assert pyinstaller == self.metadata_required()
    self = FakeGraph("import pkg_resources; pkg_resources.get_distribution('pyinstaller')")
    assert pyinstaller == self.metadata_required()
    self = FakeGraph("import pkg_resources; pkg_resources.require('pyinstaller')")
    assert with_dependencies == self.metadata_required()