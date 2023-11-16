def _pyi_rthook():
    if False:
        i = 10
        return i + 15
    import atexit
    import os
    import shutil
    import win32com
    import _pyi_rth_utils.tempfile
    supportdir = _pyi_rth_utils.tempfile.secure_mkdtemp()
    genpydir = os.path.join(supportdir, 'gen_py')
    os.makedirs(genpydir, exist_ok=True)
    atexit.register(shutil.rmtree, supportdir, ignore_errors=True)
    win32com.__gen_path__ = genpydir
    win32com.gen_py.__path__ = [genpydir]
_pyi_rthook()
del _pyi_rthook