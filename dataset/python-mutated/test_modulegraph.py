import unittest
from PyInstaller.lib.modulegraph import modulegraph
import os
import sys
import shutil
import warnings
from altgraph import Graph
from PyInstaller.compat import is_win
from PyInstaller.utils.tests import importorskip
import textwrap
import pickle
from importlib._bootstrap_external import SourceFileLoader, ExtensionFileLoader
from zipimport import zipimporter
try:
    bytes
except NameError:
    bytes = str
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
TESTDATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'testdata', 'nspkg')
READ_MODE = 'U' if sys.version_info[:2] < (3, 4) else 'r'
try:
    expectedFailure = unittest.expectedFailure
except AttributeError:
    import functools

    def expectedFailure(function):
        if False:
            while True:
                i = 10

        @functools.wraps(function)
        def wrapper(*args, **kwds):
            if False:
                while True:
                    i = 10
            try:
                function(*args, **kwds)
            except AssertionError:
                pass
            else:
                self.fail('unexpected pass')

class TestDependencyInfo(unittest.TestCase):

    def test_pickling(self):
        if False:
            print('Hello World!')
        info = modulegraph.DependencyInfo(function=True, conditional=False, tryexcept=True, fromlist=False)
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            b = pickle.dumps(info, proto)
            self.assertTrue(isinstance(b, bytes))
            o = pickle.loads(b)
            self.assertEqual(o, info)

    def test_merging(self):
        if False:
            return 10
        info1 = modulegraph.DependencyInfo(function=True, conditional=False, tryexcept=True, fromlist=False)
        info2 = modulegraph.DependencyInfo(function=False, conditional=True, tryexcept=True, fromlist=False)
        self.assertEqual(info1._merged(info2), modulegraph.DependencyInfo(function=True, conditional=True, tryexcept=True, fromlist=False))
        info2 = modulegraph.DependencyInfo(function=False, conditional=True, tryexcept=False, fromlist=False)
        self.assertEqual(info1._merged(info2), modulegraph.DependencyInfo(function=True, conditional=True, tryexcept=True, fromlist=False))
        info2 = modulegraph.DependencyInfo(function=False, conditional=False, tryexcept=False, fromlist=False)
        self.assertEqual(info1._merged(info2), modulegraph.DependencyInfo(function=False, conditional=False, tryexcept=False, fromlist=False))
        info1 = modulegraph.DependencyInfo(function=True, conditional=False, tryexcept=True, fromlist=True)
        self.assertEqual(info1._merged(info2), modulegraph.DependencyInfo(function=False, conditional=False, tryexcept=False, fromlist=False))
        info2 = modulegraph.DependencyInfo(function=False, conditional=False, tryexcept=False, fromlist=True)
        self.assertEqual(info1._merged(info2), modulegraph.DependencyInfo(function=False, conditional=False, tryexcept=False, fromlist=True))

class TestFunctions(unittest.TestCase):
    if not hasattr(unittest.TestCase, 'assertIsInstance'):

        def assertIsInstance(self, obj, types):
            if False:
                print('Hello World!')
            self.assertTrue(isinstance(obj, types), '%r is not instance of %r' % (obj, types))

    def test_eval_str_tuple(self):
        if False:
            return 10
        for v in ['()', '("hello",)', '("hello", "world")', "('hello',)", "('hello', 'world')", '(\'hello\', "world")']:
            self.assertEqual(modulegraph._eval_str_tuple(v), eval(v))
        self.assertRaises(ValueError, modulegraph._eval_str_tuple, '')
        self.assertRaises(ValueError, modulegraph._eval_str_tuple, "'a'")
        self.assertRaises(ValueError, modulegraph._eval_str_tuple, "'a', 'b'")
        self.assertRaises(ValueError, modulegraph._eval_str_tuple, "('a', ('b', 'c'))")
        self.assertRaises(ValueError, modulegraph._eval_str_tuple, '(\'a\', (\'b", \'c\'))')

    def test_os_listdir(self):
        if False:
            while True:
                i = 10
        root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'testdata')
        if is_win:
            dirname = 'C:\\Windows\\'
            filename = 'C:\\Windows\\user32.dll\\foobar'
        else:
            dirname = '/etc/'
            filename = '/etc/hosts/foobar'
        self.assertEqual(modulegraph.os_listdir(dirname), os.listdir(dirname))
        self.assertRaises(IOError, modulegraph.os_listdir, filename)
        self.assertRaises(IOError, modulegraph.os_listdir, os.path.join(root, 'test.egg', 'bar'))
        self.assertEqual(list(sorted(modulegraph.os_listdir(os.path.join(root, 'test.egg', 'foo')))), ['bar', 'bar.txt', 'baz.txt'])

    def test_code_to_file(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            code = modulegraph._code_to_file.__code__
        except AttributeError:
            code = modulegraph._code_to_file.func_code
        data = modulegraph._code_to_file(code)
        self.assertTrue(hasattr(data, 'read'))
        content = data.read()
        self.assertIsInstance(content, bytes)
        data.close()

    def test_find_module(self):
        if False:
            print('Hello World!')
        for path in ('syspath', 'syspath.zip', 'syspath.egg'):
            path = os.path.join(os.path.dirname(TESTDATA), path)
            if os.path.exists(os.path.join(path, 'mymodule.pyc')):
                os.unlink(os.path.join(path, 'mymodule.pyc'))
            modgraph = modulegraph.ModuleGraph()
            info = modgraph._find_module('mymodule', path=[path] + sys.path)
            (filename, loader) = info
            if path.endswith('.zip') or path.endswith('.egg'):
                if filename.endswith('.py'):
                    self.assertEqual(filename, os.path.join(path, 'mymodule.py'))
                    self.assertIsInstance(loader, zipimporter)
                else:
                    self.assertEqual(filename, os.path.join(path, 'mymodule.pyc'))
                    self.assertIsInstance(loader, zipimporter)
            else:
                self.assertEqual(filename, os.path.join(path, 'mymodule.py'))
                self.assertIsInstance(loader, SourceFileLoader)
            if path.endswith('.zip') or path.endswith('.egg'):
                self.assertRaises(ImportError, modgraph._find_module, 'mymodule2', path=[path] + sys.path)
            else:
                info = modgraph._find_module('mymodule2', path=[path] + sys.path)
                (filename, loader) = info
                self.assertEqual(filename, os.path.join(path, 'mymodule2.pyc'))
            info = modgraph._find_module('mypkg', path=[path] + sys.path)
            (filename, loader) = info
            self.assertTrue(loader.is_package('mypkg'))
            if path.endswith('.zip'):
                self.assertRaises(ImportError, modgraph._find_module, 'myext', path=[path] + sys.path)
            elif path.endswith('.egg'):
                pass
            else:
                info = modgraph._find_module('myext', path=[path] + sys.path)
                (filename, loader) = info
                if sys.platform == 'win32':
                    ext = '.pyd'
                else:
                    ext = '.so'
                self.assertEqual(filename, os.path.join(path, 'myext' + ext))
                self.assertIsInstance(loader, ExtensionFileLoader)

    def test_addPackage(self):
        if False:
            while True:
                i = 10
        saved = modulegraph._packagePathMap
        self.assertIsInstance(saved, dict)
        try:
            modulegraph._packagePathMap = {}
            modulegraph.addPackagePath('foo', 'a')
            self.assertEqual(modulegraph._packagePathMap, {'foo': ['a']})
            modulegraph.addPackagePath('foo', 'b')
            self.assertEqual(modulegraph._packagePathMap, {'foo': ['a', 'b']})
            modulegraph.addPackagePath('bar', 'b')
            self.assertEqual(modulegraph._packagePathMap, {'foo': ['a', 'b'], 'bar': ['b']})
        finally:
            modulegraph._packagePathMap = saved

class TestNode(unittest.TestCase):
    if not hasattr(unittest.TestCase, 'assertIsInstance'):

        def assertIsInstance(self, obj, types):
            if False:
                print('Hello World!')
            self.assertTrue(isinstance(obj, types), '%r is not instance of %r' % (obj, types))

    def testBasicAttributes(self):
        if False:
            i = 10
            return i + 15
        n = modulegraph.Node('foobar.xyz')
        self.assertEqual(n.identifier, n.graphident)
        self.assertEqual(n.identifier, 'foobar.xyz')
        self.assertEqual(n.filename, None)
        self.assertEqual(n.packagepath, None)
        self.assertEqual(n.code, None)
        self.assertEqual(n._deferred_imports, None)
        self.assertEqual(n._starimported_ignored_module_names, set())

    def test_global_attrs(self):
        if False:
            while True:
                i = 10
        n = modulegraph.Node('foobar.xyz')
        self.assertEqual(n._global_attr_names, set())
        self.assertFalse(n.is_global_attr('foo'))
        n.add_global_attr('foo')
        self.assertTrue(n.is_global_attr('foo'))
        n.remove_global_attr_if_found('foo')
        self.assertFalse(n.is_global_attr('foo'))
        n.remove_global_attr_if_found('foo')

    def test_submodules(self):
        if False:
            while True:
                i = 10
        n = modulegraph.Node('foobar.xyz')
        self.assertEqual(n._submodule_basename_to_node, {})
        sm = modulegraph.Node('bar.baz')
        self.assertFalse(n.is_submodule('bar'))
        n.add_submodule('bar', sm)
        self.assertTrue(n.is_submodule('bar'))
        self.assertIs(n.get_submodule('bar'), sm)
        self.assertRaises(KeyError, n.get_submodule, 'XXX')
        self.assertIs(n.get_submodule_or_none('XXX'), None)

    def testOrder(self):
        if False:
            while True:
                i = 10
        n1 = modulegraph.Node('n1')
        n2 = modulegraph.Node('n2')
        self.assertTrue(n1 < n2)
        self.assertFalse(n2 < n1)
        self.assertTrue(n1 <= n1)
        self.assertFalse(n1 == n2)
        self.assertTrue(n1 == n1)
        self.assertTrue(n1 != n2)
        self.assertFalse(n1 != n1)
        self.assertTrue(n2 > n1)
        self.assertFalse(n1 > n2)
        self.assertTrue(n1 >= n1)
        self.assertTrue(n2 >= n1)

    def testHashing(self):
        if False:
            return 10
        n1a = modulegraph.Node('n1')
        n1b = modulegraph.Node('n1')
        n2 = modulegraph.Node('n2')
        d = {}
        d[n1a] = 'n1'
        d[n2] = 'n2'
        self.assertEqual(d[n1b], 'n1')
        self.assertEqual(d[n2], 'n2')

    def test_infoTuple(self):
        if False:
            for i in range(10):
                print('nop')
        n = modulegraph.Node('n1')
        self.assertEqual(n.infoTuple(), ('n1',))

    def assertNoMethods(self, klass):
        if False:
            while True:
                i = 10
        d = dict(klass.__dict__)
        del d['__doc__']
        del d['__module__']
        if '__weakref__' in d:
            del d['__weakref__']
        if '__qualname__' in d:
            del d['__qualname__']
        if '__dict__' in d:
            del d['__dict__']
        if '__slotnames__' in d:
            del d['__slotnames__']
        self.assertEqual(d, {})

    def assertHasExactMethods(self, klass, *methods):
        if False:
            print('Hello World!')
        d = dict(klass.__dict__)
        del d['__doc__']
        del d['__module__']
        if '__weakref__' in d:
            del d['__weakref__']
        if '__qualname__' in d:
            del d['__qualname__']
        if '__dict__' in d:
            del d['__dict__']
        for nm in methods:
            self.assertTrue(nm in d, "%s doesn't have attribute %r" % (klass, nm))
            del d[nm]
        self.assertEqual(d, {})
    if not hasattr(unittest.TestCase, 'assertIsSubclass'):

        def assertIsSubclass(self, cls1, cls2, message=None):
            if False:
                return 10
            self.assertTrue(issubclass(cls1, cls2), message or '%r is not a subclass of %r' % (cls1, cls2))

    def test_subclasses(self):
        if False:
            return 10
        self.assertIsSubclass(modulegraph.AliasNode, modulegraph.Node)
        self.assertIsSubclass(modulegraph.Script, modulegraph.Node)
        self.assertIsSubclass(modulegraph.BadModule, modulegraph.Node)
        self.assertIsSubclass(modulegraph.ExcludedModule, modulegraph.BadModule)
        self.assertIsSubclass(modulegraph.MissingModule, modulegraph.BadModule)
        self.assertIsSubclass(modulegraph.BaseModule, modulegraph.Node)
        self.assertIsSubclass(modulegraph.BuiltinModule, modulegraph.BaseModule)
        self.assertIsSubclass(modulegraph.SourceModule, modulegraph.BaseModule)
        self.assertIsSubclass(modulegraph.CompiledModule, modulegraph.BaseModule)
        self.assertIsSubclass(modulegraph.Package, modulegraph.BaseModule)
        self.assertIsSubclass(modulegraph.Extension, modulegraph.BaseModule)
        self.assertNoMethods(modulegraph.BadModule)
        self.assertNoMethods(modulegraph.ExcludedModule)
        self.assertNoMethods(modulegraph.MissingModule)
        self.assertNoMethods(modulegraph.BuiltinModule)
        self.assertNoMethods(modulegraph.SourceModule)
        self.assertNoMethods(modulegraph.CompiledModule)
        self.assertNoMethods(modulegraph.Package)
        self.assertNoMethods(modulegraph.Extension)
        self.assertHasExactMethods(modulegraph.Script, '__init__', 'infoTuple')
        n1 = modulegraph.Node('n1')
        n1.packagepath = ['a', 'b']
        a1 = modulegraph.AliasNode('a1', n1)
        self.assertEqual(a1.graphident, 'a1')
        self.assertEqual(a1.identifier, 'n1')
        self.assertTrue(a1.packagepath is n1.packagepath)
        self.assertIs(a1._deferred_imports, None)
        self.assertIs(a1._global_attr_names, n1._global_attr_names)
        self.assertIs(a1._starimported_ignored_module_names, n1._starimported_ignored_module_names)
        self.assertIs(a1._submodule_basename_to_node, n1._submodule_basename_to_node)
        v = a1.infoTuple()
        self.assertEqual(v, ('a1', 'n1'))
        self.assertHasExactMethods(modulegraph.Script, '__init__', 'infoTuple')
        s1 = modulegraph.Script('do_import')
        self.assertEqual(s1.graphident, 'do_import')
        self.assertEqual(s1.identifier, 'do_import')
        self.assertEqual(s1.filename, 'do_import')
        v = s1.infoTuple()
        self.assertEqual(v, ('do_import',))
        self.assertHasExactMethods(modulegraph.BaseModule, '__init__', 'infoTuple')
        m1 = modulegraph.BaseModule('foo')
        self.assertEqual(m1.graphident, 'foo')
        self.assertEqual(m1.identifier, 'foo')
        self.assertEqual(m1.filename, None)
        self.assertEqual(m1.packagepath, None)
        m1 = modulegraph.BaseModule('foo', 'bar', ['a'])
        self.assertEqual(m1.graphident, 'foo')
        self.assertEqual(m1.identifier, 'foo')
        self.assertEqual(m1.filename, 'bar')
        self.assertEqual(m1.packagepath, ['a'])

class TestModuleGraph(unittest.TestCase):
    if not hasattr(unittest.TestCase, 'assertIsInstance'):

        def assertIsInstance(self, obj, types):
            if False:
                return 10
            self.assertTrue(isinstance(obj, types), '%r is not instance of %r' % (obj, types))

    def test_constructor(self):
        if False:
            for i in range(10):
                print('nop')
        o = modulegraph.ModuleGraph()
        self.assertTrue(o.path is sys.path)
        self.assertEqual(o.lazynodes, {})
        self.assertEqual(o.replace_paths, ())
        self.assertEqual(o.debug, 0)
        g = Graph.Graph()
        o = modulegraph.ModuleGraph(['a', 'b', 'c'], ['modA'], [('fromA', 'toB'), ('fromC', 'toD')], {'modA': ['modB', 'modC'], 'modC': ['modE', 'modF']}, g, 1)
        self.assertEqual(o.path, ['a', 'b', 'c'])
        self.assertEqual(o.lazynodes, {'modA': None, 'modC': ['modE', 'modF']})
        self.assertEqual(o.replace_paths, [('fromA', 'toB'), ('fromC', 'toD')])
        self.assertTrue(o.graph is g)
        self.assertEqual(o.debug, 1)

    def testImpliedReference(self):
        if False:
            for i in range(10):
                print('nop')
        graph = modulegraph.ModuleGraph()
        record = []

        def import_hook(*args):
            if False:
                for i in range(10):
                    print('nop')
            record.append(('import_hook',) + args)
            return [graph.createNode(modulegraph.Node, args[0])]

        def _safe_import_hook(*args):
            if False:
                while True:
                    i = 10
            record.append(('_safe_import_hook',) + args)
            return [graph.createNode(modulegraph.Node, args[0])]
        graph.import_hook = import_hook
        graph._safe_import_hook = _safe_import_hook
        n1 = graph.createNode(modulegraph.Node, 'n1')
        n2 = graph.createNode(modulegraph.Node, 'n2')
        graph.implyNodeReference(n1, n2)
        (outs, ins) = map(list, graph.get_edges(n1))
        self.assertEqual(outs, [n2])
        self.assertEqual(ins, [])
        self.assertEqual(record, [])
        graph.implyNodeReference(n2, 'n3')
        n3 = graph.find_node('n3')
        (outs, ins) = map(list, graph.get_edges(n2))
        self.assertEqual(outs, [n3])
        self.assertEqual(ins, [n1])
        self.assertEqual(record, [('_safe_import_hook', 'n3', n2, None)])

    @expectedFailure
    def test_findNode(self):
        if False:
            while True:
                i = 10
        self.fail('findNode')

    def test_run_script(self):
        if False:
            print('Hello World!')
        script = os.path.join(os.path.dirname(TESTDATA), 'script')
        graph = modulegraph.ModuleGraph()
        master = graph.createNode(modulegraph.Node, 'root')
        m = graph.add_script(script, master)
        self.assertEqual(list(graph.get_edges(master)[0])[0], m)
        self.assertEqual(set(graph.get_edges(m)[0]), set([graph.find_node('sys'), graph.find_node('os')]))

    @expectedFailure
    def test_import_hook(self):
        if False:
            print('Hello World!')
        self.fail('import_hook')

    def test_determine_parent(self):
        if False:
            i = 10
            return i + 15
        graph = modulegraph.ModuleGraph()
        graph.import_hook('xml.dom', None)
        for node in graph.nodes():
            if isinstance(node, modulegraph.Package):
                break
        else:
            self.fail("No package located, should have at least 'os'")
        self.assertIsInstance(node, modulegraph.Package)
        parent = graph._determine_parent(node)
        self.assertEqual(parent.identifier, node.identifier)
        self.assertEqual(parent, graph.find_node(node.identifier))
        self.assertTrue(isinstance(parent, modulegraph.Package))
        m = graph.find_node('xml')
        self.assertEqual(graph._determine_parent(m), m)
        m = graph.find_node('xml.dom')
        self.assertEqual(graph._determine_parent(m), graph.find_node('xml.dom'))

    @expectedFailure
    def test_find_head_package(self):
        if False:
            return 10
        self.fail('find_head_package')

    @expectedFailure
    def test_ensure_fromlist(self):
        if False:
            for i in range(10):
                print('nop')
        self.fail('ensure_fromlist')

    @expectedFailure
    def test_find_all_submodules(self):
        if False:
            while True:
                i = 10
        self.fail('find_all_submodules')

    @expectedFailure
    def test_import_module(self):
        if False:
            print('Hello World!')
        self.fail('import_module')

    @expectedFailure
    def test_load_module(self):
        if False:
            while True:
                i = 10
        self.fail('load_module')

    @expectedFailure
    def test_safe_import_hook(self):
        if False:
            print('Hello World!')
        self.fail('safe_import_hook')

    @expectedFailure
    def test_scan_code(self):
        if False:
            i = 10
            return i + 15
        mod = modulegraph.Node('root')
        graph = modulegraph.ModuleGraph()
        code = compile('', '<test>', 'exec', 0, False)
        graph.scan_code(code, mod)
        self.assertEqual(list(graph.nodes()), [])
        node_map = {}

        def _safe_import(name, mod, fromlist, level):
            if False:
                for i in range(10):
                    print('nop')
            if name in node_map:
                node = node_map[name]
            else:
                node = modulegraph.Node(name)
            node_map[name] = node
            return [node]
        graph = modulegraph.ModuleGraph()
        graph._safe_import_hook = _safe_import
        code = compile(textwrap.dedent('            import sys\n            import os.path\n\n            def testfunc():\n                import shutil\n            '), '<test>', 'exec', 0, False)
        graph.scan_code(code, mod)
        modules = [node.identifier for node in graph.nodes()]
        self.assertEqual(set(node_map), set(['sys', 'os.path', 'shutil']))
        self.fail('actual test needed')

    @expectedFailure
    def test_load_package(self):
        if False:
            return 10
        self.fail('load_package')

    def test_find_module(self):
        if False:
            while True:
                i = 10
        record = []

        class MockedModuleGraph(modulegraph.ModuleGraph):

            def _find_module(self, name, path, parent=None):
                if False:
                    i = 10
                    return i + 15
                if path == None:
                    path = sys.path
                record.append((name, path))
                return super(MockedModuleGraph, self)._find_module(name, path, parent)
        mockedgraph = MockedModuleGraph()
        try:
            graph = modulegraph.ModuleGraph()
            m = graph._find_module('sys', None)
            self.assertEqual(record, [])
            self.assertEqual(m, (None, modulegraph.BUILTIN_MODULE))
            xml = graph.import_hook('xml')[0]
            self.assertEqual(xml.identifier, 'xml')
            self.assertRaises(ImportError, graph._find_module, 'xml', None)
            self.assertEqual(record, [])
            m = mockedgraph._find_module('shutil', None)
            self.assertEqual(record, [('shutil', graph.path)])
            self.assertTrue(isinstance(m, tuple))
            self.assertEqual(len(m), 2)
            srcfn = shutil.__file__
            if srcfn.endswith('.pyc'):
                srcfn = srcfn[:-1]
            self.assertEqual(os.path.realpath(m[0]), os.path.realpath(srcfn))
            self.assertIsInstance(m[1], SourceFileLoader)
            m2 = graph._find_module('shutil', None)
            self.assertEqual(m[1:], m2[1:])
            record[:] = []
            m = mockedgraph._find_module('sax', xml.packagepath, xml)
            self.assertEqual(record, [('sax', xml.packagepath)])
        finally:
            pass

    @expectedFailure
    def test_create_xref(self):
        if False:
            i = 10
            return i + 15
        self.fail('create_xref')

    @expectedFailure
    def test_itergraphreport(self):
        if False:
            return 10
        self.fail('itergraphreport')

    def test_report(self):
        if False:
            for i in range(10):
                print('nop')
        graph = modulegraph.ModuleGraph()
        saved_stdout = sys.stdout
        try:
            fp = sys.stdout = StringIO()
            graph.report()
            lines = fp.getvalue().splitlines()
            fp.close()
            self.assertEqual(len(lines), 3)
            self.assertEqual(lines[0], '')
            self.assertEqual(lines[1], 'Class           Name                      File')
            self.assertEqual(lines[2], '-----           ----                      ----')
            fp = sys.stdout = StringIO()
            graph._safe_import_hook('os', None, ())
            graph._safe_import_hook('sys', None, ())
            graph._safe_import_hook('nomod', None, ())
            graph.report()
            lines = fp.getvalue().splitlines()
            fp.close()
            self.assertEqual(lines[0], '')
            self.assertEqual(lines[1], 'Class           Name                      File')
            self.assertEqual(lines[2], '-----           ----                      ----')
            expected = []
            for n in graph.iter_graph():
                if n.filename:
                    expected.append([type(n).__name__, n.identifier, n.filename])
                else:
                    expected.append([type(n).__name__, n.identifier])
            expected.sort()
            actual = [item.split() for item in lines[3:]]
            actual.sort()
            self.assertEqual(expected, actual)
        finally:
            sys.stdout = saved_stdout

    def test_graphreport(self):
        if False:
            while True:
                i = 10

        def my_iter(flatpackages='packages'):
            if False:
                i = 10
                return i + 15
            yield 'line1\n'
            yield (str(flatpackages) + '\n')
            yield 'line2\n'
        graph = modulegraph.ModuleGraph()
        graph.itergraphreport = my_iter
        fp = StringIO()
        graph.graphreport(fp)
        self.assertEqual(fp.getvalue(), 'line1\n()\nline2\n')
        fp = StringIO()
        graph.graphreport(fp, 'deps')
        self.assertEqual(fp.getvalue(), 'line1\ndeps\nline2\n')
        saved_stdout = sys.stdout
        try:
            sys.stdout = fp = StringIO()
            graph.graphreport()
            self.assertEqual(fp.getvalue(), 'line1\n()\nline2\n')
        finally:
            sys.stdout = saved_stdout

    def test_replace_paths_in_code(self):
        if False:
            for i in range(10):
                print('nop')
        join = os.path.join
        graph = modulegraph.ModuleGraph(replace_paths=[('path1', 'path2'), (join('path3', 'path5'), 'path4')])
        co = compile(textwrap.dedent('\n        [x for x in range(4)]\n        '), join('path4', 'index.py'), 'exec', 0, 1)
        co = graph._replace_paths_in_code(co)
        self.assertEqual(co.co_filename, join('path4', 'index.py'))
        co = compile(textwrap.dedent('\n        [x for x in range(4)]\n        (x for x in range(4))\n        '), join('path1', 'index.py'), 'exec', 0, 1)
        self.assertEqual(co.co_filename, join('path1', 'index.py'))
        co = graph._replace_paths_in_code(co)
        self.assertEqual(co.co_filename, join('path2', 'index.py'))
        for c in co.co_consts:
            if isinstance(c, type(co)):
                self.assertEqual(c.co_filename, join('path2', 'index.py'))
        co = compile(textwrap.dedent('\n        [x for x in range(4)]\n        '), join('path3', 'path4', 'index.py'), 'exec', 0, 1)
        co = graph._replace_paths_in_code(co)
        self.assertEqual(co.co_filename, join('path3', 'path4', 'index.py'))
        co = compile(textwrap.dedent('\n        [x for x in range(4)]\n        '), join('path3', 'path5.py'), 'exec', 0, 1)
        co = graph._replace_paths_in_code(co)
        self.assertEqual(co.co_filename, join('path3', 'path5.py'))
        co = compile(textwrap.dedent('\n        [x for x in range(4)]\n        '), join('path3', 'path5', 'index.py'), 'exec', 0, 1)
        co = graph._replace_paths_in_code(co)
        self.assertEqual(co.co_filename, join('path4', 'index.py'))

    def test_createReference(self):
        if False:
            for i in range(10):
                print('nop')
        graph = modulegraph.ModuleGraph()
        n1 = modulegraph.Node('n1')
        n2 = modulegraph.Node('n2')
        graph.addNode(n1)
        graph.addNode(n2)
        graph.add_edge(n1, n2)
        (outs, ins) = map(list, graph.get_edges(n1))
        self.assertEqual(outs, [n2])
        self.assertEqual(ins, [])
        (outs, ins) = map(list, graph.get_edges(n2))
        self.assertEqual(outs, [])
        self.assertEqual(ins, [n1])
        e = graph.graph.edge_by_node('n1', 'n2')
        self.assertIsInstance(e, int)
        self.assertEqual(graph.graph.edge_data(e), 'direct')

    @importorskip('lxml')
    def test_create_xref(self):
        if False:
            print('Hello World!')
        graph = modulegraph.ModuleGraph()
        if __file__.endswith('.py'):
            graph.add_script(__file__)
        else:
            graph.add_script(__file__[:-1])
        graph.import_hook('os')
        graph.import_hook('xml.etree')
        graph.import_hook('unittest')
        fp = StringIO()
        graph.create_xref(out=fp)
        data = fp.getvalue()
        from lxml import etree
        parser = etree.HTMLParser(recover=False)
        tree = etree.parse(StringIO(data), parser)
        assert tree is not None
        assert len(parser.error_log) == 0

    def test_itergraphreport(self):
        if False:
            while True:
                i = 10
        graph = modulegraph.ModuleGraph()
        if __file__.endswith('.py'):
            graph.add_script(__file__)
        else:
            graph.add_script(__file__[:-1])
        graph.import_hook('os')
        graph.import_hook('xml.etree')
        graph.import_hook('unittest')
        graph.import_hook('lib2to3.fixes.fix_apply')
        fp = StringIO()
        list(graph.itergraphreport())
if __name__ == '__main__':
    unittest.main()