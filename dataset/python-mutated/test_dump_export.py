"""
Test dumping apps to static assets, and exporting. The exporting
mechanism uses the dump() method, so testing either, tests the other
to to some extend. Also note that our docs is very much a test for our
export mechanism.
"""
import os
import sys
import shutil
import tempfile
import subprocess
from flexx import flx
from flexx.util.testing import run_tests_if_main, raises, skip

def setup_module():
    if False:
        for i in range(10):
            print('nop')
    flx.manager._clear_old_pending_sessions(1)
    flx.assets.__init__()
    flx.assets.associate_asset(__name__, 'foo.js', 'xx')
    flx.assets.associate_asset(__name__, 'https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.21.0/codemirror.min.js')

def teardown_module():
    if False:
        while True:
            i = 10
    flx.manager._clear_old_pending_sessions(1)
    flx.assets.__init__()

class MyExportTestApp(flx.JsComponent):
    pass

def test_dump():
    if False:
        for i in range(10):
            print('nop')
    app = flx.App(MyExportTestApp)
    d = app.dump(None, 0)
    assert len(d) == 1 and 'myexporttestapp.html' in d.keys()
    app = flx.App(MyExportTestApp)
    app.serve('')
    d = app.dump(None, 0)
    assert len(d) == 1 and 'index.html' in d.keys()
    with raises(ValueError):
        d = app.dump('', 0)
    d = app.dump('index.htm', 0)
    assert len(d) == 1 and 'index.htm' in d.keys()
    d = app.dump('index.html', 2)
    fnames = list(d.keys())
    assert len(fnames) == 6 and 'index.html' in fnames
    assert 'flexx/assets/shared/foo.js' in d
    assert 'flexx/assets/shared/flexx-core.js' in d
    assert 'flexx/assets/shared/codemirror.min.js' in d
    d = app.dump('index.html', 3)
    fnames = list(d.keys())
    assert len(fnames) == 5 and 'index.html' in fnames
    assert 'flexx/assets/shared/foo.js' in d
    assert 'flexx/assets/shared/flexx-core.js' in d
    assert 'flexx/assets/shared/codemirror.min.js' not in d

def test_export():
    if False:
        i = 10
        return i + 15
    dir = os.path.join(tempfile.gettempdir(), 'flexx_export')
    if os.path.isdir(dir):
        shutil.rmtree(dir)
    app = flx.App(MyExportTestApp)
    app.export(dir, 0)
    assert len(os.listdir(dir)) == 1
    assert os.path.isfile(os.path.join(dir, 'myexporttestapp.html'))
    app.export(dir, 3)
    assert len(os.listdir(dir)) == 2
    assert os.path.isfile(os.path.join(dir, 'flexx', 'assets', 'shared', 'reset.css'))
    assert os.path.isfile(os.path.join(dir, 'flexx', 'assets', 'shared', 'flexx-core.js'))
    assert os.path.isfile(os.path.join(dir, 'flexx', 'assets', 'shared', 'foo.js'))
    app.export(os.path.join(dir, 'foo.html'))
    assert len(os.listdir(dir)) == 3
    assert os.path.isfile(os.path.join(dir, 'foo.html'))

def test_dump_consistency():
    if False:
        i = 10
        return i + 15
    app1 = flx.App(MyExportTestApp)
    d1 = app1.dump()
    app2 = flx.App(MyExportTestApp)
    d2 = app2.dump()
    assert d1 == d2

def test_assetstore_data():
    if False:
        print('Hello World!')
    store = flx.assets.__class__()
    store.add_shared_data('foo.png', b'xx')
    d = store._dump_data()
    assert len(d) == 1 and 'flexx/data/shared/foo.png' in d.keys()
CODE = "\nimport sys\nfrom flexx import flx\n\nclass Foo(flx.Widget):\n    pass\n\napp = flx.App(Foo)\nd = app.dump()\nfor fname in ['foo.html', 'flexx/assets/shared/flexx.ui._widget.js']:\n    assert fname in d\n\nassert not flx.manager.get_app_names(), 'manager.get_app_names not empty'\nassert not flx.manager._appinfo, 'manager._appinfo not empty'\nassert 'tornado' not in sys.modules, 'tornado unexpectedly imported'\n"

def test_dump_side_effects():
    if False:
        print('Hello World!')
    p = subprocess.Popen([sys.executable, '-c', CODE], env=os.environ, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out = p.communicate()[0]
    if p.returncode:
        raise RuntimeError(out.decode())
run_tests_if_main()