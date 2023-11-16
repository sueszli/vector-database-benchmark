def _pyi_rthook():
    if False:
        i = 10
        return i + 15
    import os
    import sys
    os.environ['GTK_DATA_PREFIX'] = sys._MEIPASS
    os.environ['GTK_EXE_PREFIX'] = sys._MEIPASS
    os.environ['GTK_PATH'] = sys._MEIPASS
    os.environ['PANGO_LIBDIR'] = sys._MEIPASS
    os.environ['PANGO_SYSCONFDIR'] = os.path.join(sys._MEIPASS, 'etc')
_pyi_rthook()
del _pyi_rthook