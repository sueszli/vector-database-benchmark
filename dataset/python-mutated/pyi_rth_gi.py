def _pyi_rthook():
    if False:
        print('Hello World!')
    import os
    import sys
    os.environ['GI_TYPELIB_PATH'] = os.path.join(sys._MEIPASS, 'gi_typelibs')
_pyi_rthook()
del _pyi_rthook