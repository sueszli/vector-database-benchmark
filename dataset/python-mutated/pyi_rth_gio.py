def _pyi_rthook():
    if False:
        print('Hello World!')
    import os
    import sys
    os.environ['GIO_MODULE_DIR'] = os.path.join(sys._MEIPASS, 'gio_modules')
_pyi_rthook()
del _pyi_rthook