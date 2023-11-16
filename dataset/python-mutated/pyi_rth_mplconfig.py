def _pyi_rthook():
    if False:
        print('Hello World!')
    import atexit
    import os
    import shutil
    import _pyi_rth_utils.tempfile
    configdir = _pyi_rth_utils.tempfile.secure_mkdtemp()
    os.environ['MPLCONFIGDIR'] = configdir
    try:
        atexit.register(shutil.rmtree, configdir, ignore_errors=True)
    except OSError:
        pass
_pyi_rthook()
del _pyi_rthook