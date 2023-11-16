import ast
import os
import os.path
import sys
import py_compile
import textwrap
import zipfile
from importlib.machinery import EXTENSION_SUFFIXES
import pytest
from PyInstaller.lib.modulegraph import modulegraph
from PyInstaller.utils.tests import xfail

def _import_and_get_node(tmpdir, module_name, path=None):
    if False:
        return 10
    script = tmpdir.join('script.py')
    script.write('import %s' % module_name)
    if path is None:
        path = [str(tmpdir)]
    mg = modulegraph.ModuleGraph(path)
    mg.add_script(str(script))
    return mg.find_node(module_name)

def test_sourcefile(tmpdir):
    if False:
        print('Hello World!')
    tmpdir.join('source.py').write('###')
    node = _import_and_get_node(tmpdir, 'source')
    assert isinstance(node, modulegraph.SourceModule)

def test_invalid_sourcefile(tmpdir):
    if False:
        while True:
            i = 10
    tmpdir.join('invalid_source.py').write('invalid python-source code')
    node = _import_and_get_node(tmpdir, 'invalid_source')
    assert isinstance(node, modulegraph.InvalidSourceModule)

def test_invalid_compiledfile(tmpdir):
    if False:
        return 10
    tmpdir.join('invalid_compiled.pyc').write('invalid byte-code')
    node = _import_and_get_node(tmpdir, 'invalid_compiled')
    assert isinstance(node, modulegraph.InvalidCompiledModule)

def test_builtin(tmpdir):
    if False:
        while True:
            i = 10
    node = _import_and_get_node(tmpdir, 'sys', path=sys.path)
    assert isinstance(node, modulegraph.BuiltinModule)

def test_extension(tmpdir):
    if False:
        for i in range(10):
            print('nop')
    node = _import_and_get_node(tmpdir, '_ctypes', path=sys.path)
    assert isinstance(node, modulegraph.Extension)

def test_package(tmpdir):
    if False:
        for i in range(10):
            print('nop')
    pysrc = tmpdir.join('stuff', '__init__.py')
    pysrc.write('###', ensure=True)
    node = _import_and_get_node(tmpdir, 'stuff')
    assert node.__class__ is modulegraph.Package
    assert node.filename in (str(pysrc), str(pysrc) + 'c')
    assert node.packagepath == [pysrc.dirname]

@pytest.mark.parametrize('num, modname, expected_nodetype', ((1, 'myextpkg', modulegraph.ExtensionPackage), (2, 'myextpkg', modulegraph.ExtensionPackage), (3, 'myextpkg.other', modulegraph.Extension), (4, 'myextpkg.subpkg', modulegraph.ExtensionPackage), (5, 'myextpkg.subpkg.other', modulegraph.Extension)))
def test_package_init_is_extension(tmpdir, num, modname, expected_nodetype):
    if False:
        while True:
            i = 10

    def wt(*args):
        if False:
            print('Hello World!')
        f = tmpdir.join(*args)
        f.write_text('###', encoding='ascii')
        return f

    def create_package_files(test_case):
        if False:
            print('Hello World!')
        (tmpdir / 'myextpkg' / 'subpkg').ensure(dir=True)
        m = wt('myextpkg', '__init__' + EXTENSION_SUFFIXES[0])
        if test_case == 1:
            return m
        wt('myextpkg', '__init__.py')
        if test_case == 2:
            return m
        m = wt('myextpkg', 'other.py')
        m = wt('myextpkg', 'other' + EXTENSION_SUFFIXES[0])
        if test_case == 3:
            return m
        m = wt('myextpkg', 'subpkg', '__init__.py')
        m = wt('myextpkg', 'subpkg', '__init__' + EXTENSION_SUFFIXES[0])
        if test_case == 4:
            return m
        m = wt('myextpkg', 'subpkg', 'other.py')
        m = wt('myextpkg', 'subpkg', 'other' + EXTENSION_SUFFIXES[0])
        return m
    module_file = create_package_files(num)
    node = _import_and_get_node(tmpdir, modname)
    assert node.__class__ is expected_nodetype
    if expected_nodetype is modulegraph.ExtensionPackage:
        assert node.packagepath == [module_file.dirname]
    else:
        assert node.packagepath is None
    assert node.filename == str(module_file)

def test_relative_import_missing(tmpdir):
    if False:
        i = 10
        return i + 15
    libdir = tmpdir.join('lib')
    path = [str(libdir)]
    pkg = libdir.join('pkg')
    pkg.join('__init__.py').ensure().write('#')
    pkg.join('x', '__init__.py').ensure().write('#')
    pkg.join('x', 'y', '__init__.py').ensure().write('#')
    pkg.join('x', 'y', 'z.py').ensure().write('from . import DoesNotExist')
    script = tmpdir.join('script.py')
    script.write('import pkg.x.y.z')
    mg = modulegraph.ModuleGraph(path)
    mg.add_script(str(script))
    assert isinstance(mg.find_node('pkg.x.y.z'), modulegraph.SourceModule)
    assert isinstance(mg.find_node('pkg.x.y.DoesNotExist'), modulegraph.MissingModule)

def _zip_directory(filename, path):
    if False:
        return 10
    with zipfile.ZipFile(filename, mode='w') as zfh:
        for filename in path.visit(fil='*.py*'):
            zfh.write(str(filename), filename.relto(path))

def test_zipped_module_source(tmpdir):
    if False:
        i = 10
        return i + 15
    pysrc = tmpdir.join('stuff.py')
    pysrc.write('###', ensure=True)
    zipfilename = str(tmpdir.join('unstuff.zip'))
    _zip_directory(zipfilename, tmpdir)
    node = _import_and_get_node(tmpdir, 'stuff', path=[zipfilename])
    assert node.__class__ is modulegraph.SourceModule
    assert node.filename.startswith(os.path.join(zipfilename, 'stuff.py'))

def test_zipped_module_source_and_compiled(tmpdir):
    if False:
        for i in range(10):
            print('nop')
    pysrc = tmpdir.join('stuff.py')
    pysrc.write('###', ensure=True)
    py_compile.compile(str(pysrc))
    zipfilename = str(tmpdir.join('unstuff.zip'))
    _zip_directory(zipfilename, tmpdir)
    node = _import_and_get_node(tmpdir, 'stuff', path=[zipfilename])
    assert node.__class__ in (modulegraph.SourceModule, modulegraph.CompiledModule)
    assert node.filename.startswith(os.path.join(zipfilename, 'stuff.py'))

def _zip_package(filename, path):
    if False:
        print('Hello World!')
    with zipfile.ZipFile(filename, mode='w') as zfh:
        for filename in path.visit():
            zfh.write(str(filename), filename.relto(path.dirname))

def test_zipped_package_source(tmpdir):
    if False:
        return 10
    pysrc = tmpdir.join('stuff', '__init__.py')
    pysrc.write('###', ensure=True)
    zipfilename = str(tmpdir.join('stuff.zip'))
    _zip_package(zipfilename, tmpdir.join('stuff'))
    node = _import_and_get_node(tmpdir, 'stuff', path=[zipfilename])
    assert node.__class__ is modulegraph.Package
    assert node.packagepath == [os.path.join(zipfilename, 'stuff')]

def test_zipped_package_source_and_compiled(tmpdir):
    if False:
        for i in range(10):
            print('nop')
    pysrc = tmpdir.join('stuff', '__init__.py')
    pysrc.write('###', ensure=True)
    py_compile.compile(str(pysrc))
    zipfilename = str(tmpdir.join('stuff.zip'))
    _zip_package(zipfilename, tmpdir.join('stuff'))
    node = _import_and_get_node(tmpdir, 'stuff', path=[zipfilename])
    assert node.__class__ is modulegraph.Package
    assert node.packagepath == [os.path.join(zipfilename, 'stuff')]

def test_nspackage_pep420(tmpdir):
    if False:
        return 10
    p1 = tmpdir.join('p1')
    p2 = tmpdir.join('p2')
    p1.join('stuff', 'a.py').ensure().write('###')
    p2.join('stuff', 'b.py').ensure().write('###')
    path = [str(p1), str(p2)]
    script = tmpdir.join('script.py')
    script.write('import stuff.a, stuff.b')
    mg = modulegraph.ModuleGraph(path)
    mg.add_script(str(script))
    mg.report()
    assert isinstance(mg.find_node('stuff.a'), modulegraph.SourceModule)
    assert isinstance(mg.find_node('stuff.b'), modulegraph.SourceModule)
    node = mg.find_node('stuff')
    assert isinstance(node, modulegraph.NamespacePackage)
    assert node.packagepath == [os.path.join(p, 'stuff') for p in path]

@pytest.mark.darwin
@pytest.mark.linux
def test_symlinks(tmpdir):
    if False:
        return 10
    base_dir = tmpdir.join('base').ensure(dir=True)
    p1_init = tmpdir.join('p1', '__init__.py').ensure()
    p2_init = tmpdir.join('p2', '__init__.py').ensure()
    p1_init.write('###')
    p2_init.write('###')
    base_dir.join('p1').ensure(dir=True)
    os.symlink(str(p1_init), str(base_dir.join('p1', '__init__.py')))
    os.symlink(str(p2_init), str(base_dir.join('p1', 'p2.py')))
    node = _import_and_get_node(base_dir, 'p1.p2')
    assert isinstance(node, modulegraph.SourceModule)

def test_import_order_1(tmpdir):
    if False:
        print('Hello World!')

    class MyModuleGraph(modulegraph.ModuleGraph):

        def _load_module(self, fqname, pathname, loader):
            if False:
                i = 10
                return i + 15
            if not record or record[-1] != fqname:
                record.append(fqname)
            return super()._load_module(fqname, pathname, loader)
    record = []
    for (filename, content) in (('a/', 'from . import c, d'), ('a/c', '#'), ('a/d/', 'from . import f, g, h'), ('a/d/f/', 'from . import j, k'), ('a/d/f/j', '#'), ('a/d/f/k', '#'), ('a/d/g/', 'from . import l, m'), ('a/d/g/l', '#'), ('a/d/g/m', '#'), ('a/d/h', '#'), ('b/', 'from . import e'), ('b/e/', 'from . import i'), ('b/e/i', '#')):
        if filename.endswith('/'):
            filename += '__init__'
        tmpdir.join(*(filename + '.py').split('/')).ensure().write(content)
    script = tmpdir.join('script.py')
    script.write('import a, b')
    mg = MyModuleGraph([str(tmpdir)])
    mg.add_script(str(script))
    expected = ['a', 'a.c', 'a.d', 'a.d.f', 'a.d.f.j', 'a.d.f.k', 'a.d.g', 'a.d.g.l', 'a.d.g.m', 'a.d.h', 'b', 'b.e', 'b.e.i']
    assert record == expected

def test_import_order_2(tmpdir):
    if False:
        print('Hello World!')

    class MyModuleGraph(modulegraph.ModuleGraph):

        def _load_module(self, fqname, pathname, loader):
            if False:
                while True:
                    i = 10
            if not record or record[-1] != fqname:
                record.append(fqname)
            return super()._load_module(fqname, pathname, loader)
    record = []
    for (filename, content) in (('a/', '#'), ('a/c/', '#'), ('a/c/g', '#'), ('a/c/h', 'from . import g'), ('a/d/', '#'), ('a/d/i', 'from ..c import h'), ('a/d/j/', 'from .. import i'), ('a/d/j/o', '#'), ('b/', 'from .e import k'), ('b/e/', 'import a.c.g'), ('b/e/k', 'from .. import f'), ('b/e/l', 'import a.d.j'), ('b/f/', '#'), ('b/f/m', '#'), ('b/f/n/', '#'), ('b/f/n/p', 'from ...e import l')):
        if filename.endswith('/'):
            filename += '__init__'
        tmpdir.join(*(filename + '.py').split('/')).ensure().write(content)
    script = tmpdir.join('script.py')
    script.write('import b.f.n.p')
    mg = MyModuleGraph([str(tmpdir)])
    mg.add_script(str(script))
    expected = ['b', 'b.e', 'a', 'a.c', 'a.c.g', 'b.e.k', 'b.f', 'b.f.n', 'b.f.n.p', 'b.e.l', 'a.d', 'a.d.j', 'a.d.i', 'a.c.h']
    assert record == expected
    print(record)

def __scan_code(code, use_ast, monkeypatch):
    if False:
        while True:
            i = 10
    mg = modulegraph.ModuleGraph()
    monkeypatch.setattr(mg, '_process_imports', lambda m: None)
    module = mg.createNode(modulegraph.Script, 'dummy.py')
    code = textwrap.dedent(code)
    if use_ast:
        co_ast = compile(code, 'dummy', 'exec', ast.PyCF_ONLY_AST)
        co = compile(co_ast, 'dummy', 'exec')
    else:
        co_ast = None
        co = compile(code, 'dummy', 'exec')
    mg._scan_code(module, co)
    return module

@pytest.mark.parametrize('use_ast', (True, False))
def test_scan_code__empty(monkeypatch, use_ast):
    if False:
        for i in range(10):
            print('nop')
    code = '# empty code'
    module = __scan_code(code, use_ast, monkeypatch)
    assert len(module._deferred_imports) == 0
    assert len(module._global_attr_names) == 0

@pytest.mark.parametrize('use_ast', (True, False))
def test_scan_code__basic(monkeypatch, use_ast):
    if False:
        print('Hello World!')
    code = '\n    import os.path\n    from sys import maxint, exitfunc, platform\n    del exitfunc\n    def testfunc():\n        import shutil\n    '
    module = __scan_code(code, use_ast, monkeypatch)
    assert len(module._deferred_imports) == 3
    assert [di[1][0] for di in module._deferred_imports] == ['os.path', 'sys', 'shutil']
    assert module.is_global_attr('maxint')
    assert module.is_global_attr('os')
    assert module.is_global_attr('platform')
    assert not module.is_global_attr('shutil')
    assert not module.is_global_attr('exitfunc')

def test_swig_import_simple_BUGGY(tmpdir):
    if False:
        return 10
    libdir = tmpdir.join('lib')
    path = [str(libdir)]
    osgeo = libdir.join('pyi_test_osgeo')
    osgeo.join('__init__.py').ensure().write('#')
    osgeo.join('pyi_gdal.py').write('# automatically generated by SWIG\nimport _pyi_gdal')
    osgeo.join('_pyi_gdal.py').write('#')
    script = tmpdir.join('script.py')
    script.write('from pyi_test_osgeo import pyi_gdal')
    mg = modulegraph.ModuleGraph(path)
    mg.add_script(str(script))
    assert isinstance(mg.find_node('pyi_test_osgeo'), modulegraph.Package)
    assert isinstance(mg.find_node('pyi_test_osgeo.pyi_gdal'), modulegraph.SourceModule)
    assert isinstance(mg.find_node('pyi_test_osgeo._pyi_gdal'), modulegraph.SourceModule)
    assert mg.find_node('pyi_test_osgeo._pyi_gdal').identifier == '_pyi_gdal'
    assert mg.find_node('_pyi_gdal') is None
    return mg

@xfail
def test_swig_import_simple(tmpdir):
    if False:
        print('Hello World!')
    mg = test_swig_import_simple_BUGGY(tmpdir)
    assert mg.find_node('pyi_test_osgeo._pyi_gdal') is None
    assert isinstance(mg.find_node('_pyi_gdal'), modulegraph.SourceModule)

def test_swig_import_from_top_level(tmpdir):
    if False:
        i = 10
        return i + 15
    libdir = tmpdir.join('lib')
    path = [str(libdir)]
    osgeo = libdir.join('pyi_test_osgeo')
    osgeo.join('__init__.py').ensure().write('import _pyi_gdal')
    osgeo.join('pyi_gdal.py').write('# automatically generated by SWIG\nimport _pyi_gdal')
    osgeo.join('_pyi_gdal.py').write('#')
    script = tmpdir.join('script.py')
    script.write('from pyi_test_osgeo import pyi_gdal')
    mg = modulegraph.ModuleGraph(path)
    mg.add_script(str(script))
    assert isinstance(mg.find_node('pyi_test_osgeo'), modulegraph.Package)
    assert isinstance(mg.find_node('pyi_test_osgeo.pyi_gdal'), modulegraph.SourceModule)
    assert isinstance(mg.find_node('pyi_test_osgeo._pyi_gdal'), modulegraph.SourceModule)
    assert mg.find_node('_pyi_gdal') is None

def test_swig_import_from_top_level_missing(tmpdir):
    if False:
        for i in range(10):
            print('nop')
    libdir = tmpdir.join('lib')
    path = [str(libdir)]
    osgeo = libdir.join('pyi_test_osgeo')
    osgeo.join('__init__.py').ensure().write('import _pyi_gdal')
    osgeo.join('pyi_gdal.py').write('# automatically generated by SWIG\nimport _pyi_gdal')
    script = tmpdir.join('script.py')
    script.write('from pyi_test_osgeo import pyi_gdal')
    mg = modulegraph.ModuleGraph(path)
    mg.add_script(str(script))
    assert isinstance(mg.find_node('pyi_test_osgeo'), modulegraph.Package)
    assert isinstance(mg.find_node('pyi_test_osgeo.pyi_gdal'), modulegraph.SourceModule)
    assert isinstance(mg.find_node('pyi_test_osgeo._pyi_gdal'), modulegraph.MissingModule)
    assert mg.find_node('_pyi_gdal') is None

def test_swig_import_from_top_level_but_nested(tmpdir):
    if False:
        while True:
            i = 10
    libdir = tmpdir.join('lib')
    path = [str(libdir)]
    osgeo = libdir.join('pyi_test_osgeo')
    osgeo.join('__init__.py').ensure().write('#')
    osgeo.join('x', '__init__.py').ensure().write('#')
    osgeo.join('x', 'y', '__init__.py').ensure().write('import _pyi_gdal')
    osgeo.join('x', 'y', 'pyi_gdal.py').write('# automatically generated by SWIG\nimport _pyi_gdal')
    osgeo.join('x', 'y', '_pyi_gdal.py').write('#')
    script = tmpdir.join('script.py')
    script.write('from pyi_test_osgeo.x.y import pyi_gdal')
    mg = modulegraph.ModuleGraph(path)
    mg.add_script(str(script))
    assert isinstance(mg.find_node('pyi_test_osgeo.x.y.pyi_gdal'), modulegraph.SourceModule)
    assert isinstance(mg.find_node('pyi_test_osgeo.x.y._pyi_gdal'), modulegraph.SourceModule)
    assert mg.find_node('_pyi_gdal') is None

def test_swig_top_level_but_no_swig_at_all(tmpdir):
    if False:
        while True:
            i = 10
    libdir = tmpdir.join('lib')
    path = [str(libdir)]
    libdir.join('pyi_dezimal.py').ensure().write('import _pyi_dezimal')
    script = tmpdir.join('script.py')
    script.write('import pyi_dezimal')
    mg = modulegraph.ModuleGraph(path)
    mg.add_script(str(script))
    assert isinstance(mg.find_node('pyi_dezimal'), modulegraph.SourceModule)
    assert isinstance(mg.find_node('_pyi_dezimal'), modulegraph.MissingModule)

def test_swig_top_level_but_no_swig_at_all_existing(tmpdir):
    if False:
        print('Hello World!')
    libdir = tmpdir.join('lib')
    path = [str(libdir)]
    libdir.join('pyi_dezimal.py').ensure().write('import _pyi_dezimal')
    libdir.join('_pyi_dezimal.py').ensure().write('#')
    script = tmpdir.join('script.py')
    script.write('import pyi_dezimal')
    mg = modulegraph.ModuleGraph(path)
    mg.add_script(str(script))
    assert isinstance(mg.find_node('pyi_dezimal'), modulegraph.SourceModule)
    assert isinstance(mg.find_node('_pyi_dezimal'), modulegraph.SourceModule)

def test_swig_candidate_but_not_swig(tmpdir):
    if False:
        print('Hello World!')
    libdir = tmpdir.join('lib')
    path = [str(libdir)]
    pkg = libdir.join('pkg')
    pkg.join('__init__.py').ensure().write('from . import mymod')
    pkg.join('mymod.py').write('import _mymod')
    pkg.join('_mymod.py').write('#')
    script = tmpdir.join('script.py')
    script.write('from pkg import XXX')
    mg = modulegraph.ModuleGraph(path)
    mg.add_script(str(script))
    assert isinstance(mg.find_node('pkg'), modulegraph.Package)
    assert isinstance(mg.find_node('pkg.mymod'), modulegraph.SourceModule)
    assert mg.find_node('pkg._mymod') is None
    assert isinstance(mg.find_node('_mymod'), modulegraph.MissingModule)

def test_swig_candidate_but_not_swig2(tmpdir):
    if False:
        for i in range(10):
            print('nop')
    '\n    Variation of test_swig_candidate_but_not_swig using different import statements\n    (like tifffile/tifffile.py does).\n    '
    libdir = tmpdir.join('lib')
    path = [str(libdir)]
    pkg = libdir.join('pkg')
    pkg.join('__init__.py').ensure().write('from . import mymod')
    pkg.join('mymod.py').write('from . import _mymod\nimport _mymod')
    pkg.join('_mymod.py').write('#')
    script = tmpdir.join('script.py')
    script.write('from pkg import XXX')
    mg = modulegraph.ModuleGraph(path)
    mg.add_script(str(script))
    assert isinstance(mg.find_node('pkg'), modulegraph.Package)
    assert isinstance(mg.find_node('pkg.mymod'), modulegraph.SourceModule)
    assert isinstance(mg.find_node('pkg._mymod'), modulegraph.SourceModule)
    assert isinstance(mg.find_node('_mymod'), modulegraph.MissingModule)

def test_swig_candidate_but_not_swig_missing(tmpdir):
    if False:
        return 10
    libdir = tmpdir.join('lib')
    path = [str(libdir)]
    pkg = libdir.join('pkg')
    pkg.join('__init__.py').ensure().write('from . import mymod')
    pkg.join('mymod.py').write('import _mymod')
    script = tmpdir.join('script.py')
    script.write('import pkg')
    mg = modulegraph.ModuleGraph(path)
    mg.add_script(str(script))
    assert isinstance(mg.find_node('pkg'), modulegraph.Package)
    assert isinstance(mg.find_node('pkg.mymod'), modulegraph.SourceModule)
    assert mg.find_node('pkg._mymod') is None
    assert isinstance(mg.find_node('_mymod'), modulegraph.MissingModule)

def test_swig_candidate_but_not_swig_missing2(tmpdir):
    if False:
        print('Hello World!')
    '\n    Variation of test_swig_candidate_but_not_swig_missing using different import statements\n    (like tifffile/tifffile.py does).\n    '
    libdir = tmpdir.join('lib')
    path = [str(libdir)]
    pkg = libdir.join('pkg')
    pkg.join('__init__.py').ensure().write('from . import mymod')
    pkg.join('mymod.py').write('from . import _mymod\nimport _mymod')
    script = tmpdir.join('script.py')
    script.write('import pkg')
    mg = modulegraph.ModuleGraph(path)
    mg.add_script(str(script))
    assert isinstance(mg.find_node('pkg'), modulegraph.Package)
    assert isinstance(mg.find_node('pkg.mymod'), modulegraph.SourceModule)
    assert isinstance(mg.find_node('pkg._mymod'), modulegraph.MissingModule)
    assert isinstance(mg.find_node('_mymod'), modulegraph.MissingModule)