from __future__ import absolute_import, print_function
from pyximport import pyximport
pyximport.install(reload_support=True)
import os
import shutil
import sys
import tempfile
import time
from zipfile import ZipFile
try:
    from __builtin__ import reload
except ImportError:
    from importlib import reload

def make_tempdir():
    if False:
        i = 10
        return i + 15
    tempdir = os.path.join(tempfile.gettempdir(), 'pyrex_temp')
    if os.path.exists(tempdir):
        remove_tempdir(tempdir)
    os.mkdir(tempdir)
    return tempdir

def remove_tempdir(tempdir):
    if False:
        print('Hello World!')
    shutil.rmtree(tempdir, 0, on_remove_file_error)

def on_remove_file_error(func, path, excinfo):
    if False:
        return 10
    print('Sorry! Could not remove a temp file:', path)
    print('Extra information.')
    print(func, excinfo)
    print('You may want to delete this yourself when you get a chance.')

def test_with_reload():
    if False:
        for i in range(10):
            print('nop')
    pyximport._test_files = []
    tempdir = make_tempdir()
    sys.path.append(tempdir)
    filename = os.path.join(tempdir, 'dummy.pyx')
    with open(filename, 'w') as fid:
        fid.write("print 'Hello world from the Pyrex install hook'")
    import dummy
    reload(dummy)
    depend_filename = os.path.join(tempdir, 'dummy.pyxdep')
    with open(depend_filename, 'w') as depend_file:
        depend_file.write('*.txt\nfoo.bar')
    build_filename = os.path.join(tempdir, 'dummy.pyxbld')
    with open(build_filename, 'w') as build_file:
        build_file.write('\nfrom distutils.extension import Extension\ndef make_ext(name, filename):\n    return Extension(name=name, sources=[filename])\n')
    with open(os.path.join(tempdir, 'foo.bar'), 'w') as fid:
        fid.write(' ')
    with open(os.path.join(tempdir, '1.txt'), 'w') as fid:
        fid.write(' ')
    with open(os.path.join(tempdir, 'abc.txt'), 'w') as fid:
        fid.write(' ')
    reload(dummy)
    assert len(pyximport._test_files) == 1, pyximport._test_files
    reload(dummy)
    time.sleep(1)
    with open(os.path.join(tempdir, 'abc.txt'), 'w') as fid:
        fid.write(' ')
    print('Here goes the reload')
    reload(dummy)
    assert len(pyximport._test_files) == 1, pyximport._test_files
    reload(dummy)
    assert len(pyximport._test_files) == 0, pyximport._test_files
    remove_tempdir(tempdir)

def test_zip():
    if False:
        print('Hello World!')
    try:
        import test_zip_module
    except ImportError:
        pass
    else:
        assert False, 'test_zip_module already exists'
    (fd, zip_path) = tempfile.mkstemp(suffix='.zip')
    os.close(fd)
    try:
        with ZipFile(zip_path, 'w') as zf:
            zf.writestr('test_zip_module.pyx', b'x = 42')
        sys.path.insert(0, zip_path)
        import test_zip_module
        assert test_zip_module.x == 42
    finally:
        if zip_path in sys.path:
            sys.path.remove(zip_path)
        os.remove(zip_path)

def test_zip_nonexisting():
    if False:
        i = 10
        return i + 15
    sys.path.append('nonexisting_zip_module.zip')
    try:
        import nonexisting_zip_module
    except ImportError:
        pass
    finally:
        sys.path.remove('nonexisting_zip_module.zip')
if __name__ == '__main__':
    test_with_reload()
    test_zip()
    test_zip_nonexisting()