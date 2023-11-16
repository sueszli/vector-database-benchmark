import os
from numpy.distutils.npy_pkg_config import read_config, parse_flags
from numpy.testing import temppath, assert_
simple = '[meta]\nName = foo\nDescription = foo lib\nVersion = 0.1\n\n[default]\ncflags = -I/usr/include\nlibs = -L/usr/lib\n'
simple_d = {'cflags': '-I/usr/include', 'libflags': '-L/usr/lib', 'version': '0.1', 'name': 'foo'}
simple_variable = '[meta]\nName = foo\nDescription = foo lib\nVersion = 0.1\n\n[variables]\nprefix = /foo/bar\nlibdir = ${prefix}/lib\nincludedir = ${prefix}/include\n\n[default]\ncflags = -I${includedir}\nlibs = -L${libdir}\n'
simple_variable_d = {'cflags': '-I/foo/bar/include', 'libflags': '-L/foo/bar/lib', 'version': '0.1', 'name': 'foo'}

class TestLibraryInfo:

    def test_simple(self):
        if False:
            for i in range(10):
                print('nop')
        with temppath('foo.ini') as path:
            with open(path, 'w') as f:
                f.write(simple)
            pkg = os.path.splitext(path)[0]
            out = read_config(pkg)
        assert_(out.cflags() == simple_d['cflags'])
        assert_(out.libs() == simple_d['libflags'])
        assert_(out.name == simple_d['name'])
        assert_(out.version == simple_d['version'])

    def test_simple_variable(self):
        if False:
            while True:
                i = 10
        with temppath('foo.ini') as path:
            with open(path, 'w') as f:
                f.write(simple_variable)
            pkg = os.path.splitext(path)[0]
            out = read_config(pkg)
        assert_(out.cflags() == simple_variable_d['cflags'])
        assert_(out.libs() == simple_variable_d['libflags'])
        assert_(out.name == simple_variable_d['name'])
        assert_(out.version == simple_variable_d['version'])
        out.vars['prefix'] = '/Users/david'
        assert_(out.cflags() == '-I/Users/david/include')

class TestParseFlags:

    def test_simple_cflags(self):
        if False:
            i = 10
            return i + 15
        d = parse_flags('-I/usr/include')
        assert_(d['include_dirs'] == ['/usr/include'])
        d = parse_flags('-I/usr/include -DFOO')
        assert_(d['include_dirs'] == ['/usr/include'])
        assert_(d['macros'] == ['FOO'])
        d = parse_flags('-I /usr/include -DFOO')
        assert_(d['include_dirs'] == ['/usr/include'])
        assert_(d['macros'] == ['FOO'])

    def test_simple_lflags(self):
        if False:
            print('Hello World!')
        d = parse_flags('-L/usr/lib -lfoo -L/usr/lib -lbar')
        assert_(d['library_dirs'] == ['/usr/lib', '/usr/lib'])
        assert_(d['libraries'] == ['foo', 'bar'])
        d = parse_flags('-L /usr/lib -lfoo -L/usr/lib -lbar')
        assert_(d['library_dirs'] == ['/usr/lib', '/usr/lib'])
        assert_(d['libraries'] == ['foo', 'bar'])