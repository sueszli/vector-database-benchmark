"""Test inventory util functions."""
import os
import posixpath
import zlib
from io import BytesIO
from sphinx.testing.util import SphinxTestApp
from sphinx.util.inventory import InventoryFile
inventory_v1 = b'# Sphinx inventory version 1\n# Project: foo\n# Version: 1.0\nmodule mod foo.html\nmodule.cls class foo.html\n'
inventory_v2 = b'# Sphinx inventory version 2\n# Project: foo\n# Version: 2.0\n# The remainder of this file is compressed with zlib.\n' + zlib.compress(b'module1 py:module 0 foo.html#module-module1 Long Module desc\nmodule2 py:module 0 foo.html#module-$ -\nmodule1.func py:function 1 sub/foo.html#$ -\nmodule1.Foo.bar py:method 1 index.html#foo.Bar.baz -\nCFunc c:function 2 cfunc.html#CFunc -\nstd cpp:type 1 index.html#std -\nstd::uint8_t cpp:type 1 index.html#std_uint8_t -\nfoo::Bar cpp:class 1 index.html#cpp_foo_bar -\nfoo::Bar::baz cpp:function 1 index.html#cpp_foo_bar_baz -\nfoons cpp:type 1 index.html#foons -\nfoons::bartype cpp:type 1 index.html#foons_bartype -\na term std:term -1 glossary.html#term-a-term -\nls.-l std:cmdoption 1 index.html#cmdoption-ls-l -\ndocname std:doc -1 docname.html -\nfoo js:module 1 index.html#foo -\nfoo.bar js:class 1 index.html#foo.bar -\nfoo.bar.baz js:method 1 index.html#foo.bar.baz -\nfoo.bar.qux js:data 1 index.html#foo.bar.qux -\na term including:colon std:term -1 glossary.html#term-a-term-including-colon -\n')
inventory_v2_not_having_version = b'# Sphinx inventory version 2\n# Project: foo\n# Version:\n# The remainder of this file is compressed with zlib.\n' + zlib.compress(b'module1 py:module 0 foo.html#module-module1 Long Module desc\n')

def test_read_inventory_v1():
    if False:
        i = 10
        return i + 15
    f = BytesIO(inventory_v1)
    invdata = InventoryFile.load(f, '/util', posixpath.join)
    assert invdata['py:module']['module'] == ('foo', '1.0', '/util/foo.html#module-module', '-')
    assert invdata['py:class']['module.cls'] == ('foo', '1.0', '/util/foo.html#module.cls', '-')

def test_read_inventory_v2():
    if False:
        while True:
            i = 10
    f = BytesIO(inventory_v2)
    invdata = InventoryFile.load(f, '/util', posixpath.join)
    assert len(invdata['py:module']) == 2
    assert invdata['py:module']['module1'] == ('foo', '2.0', '/util/foo.html#module-module1', 'Long Module desc')
    assert invdata['py:module']['module2'] == ('foo', '2.0', '/util/foo.html#module-module2', '-')
    assert invdata['py:function']['module1.func'][2] == '/util/sub/foo.html#module1.func'
    assert invdata['c:function']['CFunc'][2] == '/util/cfunc.html#CFunc'
    assert invdata['std:term']['a term'][2] == '/util/glossary.html#term-a-term'
    assert invdata['std:term']['a term including:colon'][2] == '/util/glossary.html#term-a-term-including-colon'

def test_read_inventory_v2_not_having_version():
    if False:
        while True:
            i = 10
    f = BytesIO(inventory_v2_not_having_version)
    invdata = InventoryFile.load(f, '/util', posixpath.join)
    assert invdata['py:module']['module1'] == ('foo', '', '/util/foo.html#module-module1', 'Long Module desc')

def _write_appconfig(dir, language, prefix=None):
    if False:
        print('Hello World!')
    prefix = prefix or language
    os.makedirs(dir / prefix, exist_ok=True)
    (dir / prefix / 'conf.py').write_text(f'language = "{language}"', encoding='utf8')
    (dir / prefix / 'index.rst').write_text('index.rst', encoding='utf8')
    assert sorted(os.listdir(dir / prefix)) == ['conf.py', 'index.rst']
    assert (dir / prefix / 'index.rst').exists()
    return dir / prefix

def _build_inventory(srcdir):
    if False:
        i = 10
        return i + 15
    app = SphinxTestApp(srcdir=srcdir)
    app.build()
    app.cleanup()
    return app.outdir / 'objects.inv'

def test_inventory_localization(tmp_path):
    if False:
        for i in range(10):
            print('nop')
    srcdir_et = _write_appconfig(tmp_path, 'et')
    inventory_et = _build_inventory(srcdir_et)
    srcdir_en = _write_appconfig(tmp_path, 'en')
    inventory_en = _build_inventory(srcdir_en)
    assert inventory_et.read_bytes() != inventory_en.read_bytes()