import pytest
from pytest_pyodide import run_in_pyodide

def test_threading_import(selenium):
    if False:
        print('Hello World!')
    selenium.run('\n        from threading import Thread\n        ')
    selenium.run('\n        from threading import RLock\n\n        with RLock():\n            pass\n        ')
    selenium.run('\n        from threading import Lock\n\n        with Lock():\n            pass\n        ')
    selenium.run('\n        import threading\n        threading.local(); pass\n        ')
    msg = "can't start new thread"
    with pytest.raises(selenium.JavascriptException, match=msg):
        selenium.run('\n            from threading import Thread\n\n            def set_state():\n                return\n            th = Thread(target=set_state)\n            th.start()\n            ')

@run_in_pyodide
def test_multiprocessing(selenium):
    if False:
        i = 10
        return i + 15
    import multiprocessing
    from multiprocessing import connection, cpu_count
    import pytest
    res = cpu_count()
    assert isinstance(res, int)
    assert res > 0
    from multiprocessing import Process

    def func():
        if False:
            for i in range(10):
                print('nop')
        return
    process = Process(target=func)
    with pytest.raises(OSError, match='Function not implemented'):
        process.start()

@pytest.mark.requires_dynamic_linking
@run_in_pyodide
def test_ctypes_util_find_library(selenium):
    if False:
        i = 10
        return i + 15
    import os
    from ctypes.util import find_library
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, 'libfoo.so'), 'wb') as f:
            f.write(b'\x00asm\x01\x00\x00\x00\x00\x08\x04name\x02\x01\x00')
        with open(os.path.join(tmpdir, 'libbar.so'), 'wb') as f:
            f.write(b'\x00asm\x01\x00\x00\x00\x00\x08\x04name\x02\x01\x00')
        os.environ['LD_LIBRARY_PATH'] = tmpdir
        assert find_library('foo') == os.path.join(tmpdir, 'libfoo.so')
        assert find_library('bar') == os.path.join(tmpdir, 'libbar.so')
        assert find_library('baz') is None

@run_in_pyodide
def test_encodings_deepfrozen(selenium):
    if False:
        for i in range(10):
            print('nop')
    import codecs
    import encodings
    import encodings.aliases
    import encodings.ascii
    import encodings.cp437
    import encodings.utf_8
    modules = [encodings, encodings.utf_8, encodings.aliases, encodings.cp437, encodings.ascii]
    for mod in modules:
        assert 'frozen' not in repr(mod)
    all_encodings = ['ascii', 'base64_codec', 'big5', 'big5hkscs', 'bz2_codec', 'charmap', 'cp037', 'cp1006', 'cp1026', 'cp1125', 'cp1140', 'cp1250', 'cp1251', 'cp1252', 'cp1253', 'cp1254', 'cp1255', 'cp1256', 'cp1257', 'cp1258', 'cp273', 'cp424', 'cp437', 'cp500', 'cp720', 'cp737', 'cp775', 'cp850', 'cp852', 'cp855', 'cp856', 'cp857', 'cp858', 'cp860', 'cp861', 'cp862', 'cp863', 'cp864', 'cp865', 'cp866', 'cp869', 'cp874', 'cp875', 'cp932', 'cp949', 'cp950', 'euc_jis_2004', 'euc_jisx0213', 'euc_jp', 'euc_kr', 'gb18030', 'gb2312', 'gbk', 'hex_codec', 'hp_roman8', 'hz', 'idna', 'iso2022_jp', 'iso2022_jp_1', 'iso2022_jp_2', 'iso2022_jp_2004', 'iso2022_jp_3', 'iso2022_jp_ext', 'iso2022_kr', 'iso8859_1', 'iso8859_10', 'iso8859_11', 'iso8859_13', 'iso8859_14', 'iso8859_15', 'iso8859_16', 'iso8859_2', 'iso8859_3', 'iso8859_4', 'iso8859_5', 'iso8859_6', 'iso8859_7', 'iso8859_8', 'iso8859_9', 'johab', 'koi8_r', 'koi8_t', 'koi8_u', 'kz1048', 'latin_1', 'mac_arabic', 'mac_croatian', 'mac_cyrillic', 'mac_farsi', 'mac_greek', 'mac_iceland', 'mac_latin2', 'mac_roman', 'mac_romanian', 'mac_turkish', 'palmos', 'ptcp154', 'punycode', 'quopri_codec', 'raw_unicode_escape', 'rot_13', 'shift_jis', 'shift_jis_2004', 'shift_jisx0213', 'tis_620', 'undefined', 'unicode_escape', 'utf_16', 'utf_16_be', 'utf_16_le', 'utf_32', 'utf_32_be', 'utf_32_le', 'utf_7', 'utf_8', 'utf_8_sig', 'uu_codec', 'zlib_codec']
    for enc in all_encodings:
        codecs.getencoder(enc)
        codecs.getdecoder(enc)

@run_in_pyodide
def test_zipimport_traceback(selenium):
    if False:
        print('Hello World!')
    '\n    Test that traceback of modules loaded from zip file are shown as intended.\n\n    For .py files, the traceback should show the path to the .py file in the\n    zip file, e.g. "/lib/python311.zip/path/to/module.py".\n\n    For .pyc files (TODO), the traceback only shows filename, e.g. "module.py".\n    '
    import json.decoder
    import pathlib
    import sys
    import traceback
    zipfile = f'python{sys.version_info[0]}{sys.version_info[1]}.zip'
    try:
        pathlib.Path('not/exists').write_text('hello')
    except Exception:
        (_, _, exc_traceback) = sys.exc_info()
        tb = traceback.extract_tb(exc_traceback)
        assert zipfile in tb[-1].filename.split('/')
        assert tb[-1].filename == pathlib.__file__
    try:
        json.decoder.JSONDecoder().decode(1)
    except Exception:
        (_, _, exc_traceback) = sys.exc_info()
        tb = traceback.extract_tb(exc_traceback)
        assert zipfile in tb[-1].filename.split('/')
        assert tb[-1].filename == json.decoder.__file__

@run_in_pyodide
def test_zipimport_check_non_stdlib(selenium):
    if False:
        return 10
    '\n    Check if unwanted modules are included in the zip file.\n    '
    import pathlib
    import shutil
    import sys
    import tempfile
    extra_files = {'LICENSE.txt', '__phello__', '__hello__', '_sysconfigdata__emscripten_wasm32-emscripten', 'site-packages', 'lib-dynload', 'pyodide', '_pyodide'}
    stdlib_names = sys.stdlib_module_names | extra_files
    zipfile = pathlib.Path(shutil.__file__).parent
    tmpdir = pathlib.Path(tempfile.mkdtemp())
    shutil.unpack_archive(zipfile, tmpdir, 'zip')
    for f in tmpdir.glob('*'):
        assert f.name.removesuffix('.py') in stdlib_names, f.name