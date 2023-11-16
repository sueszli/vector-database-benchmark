def _pyi_rthook():
    if False:
        return 10
    import os
    import sys
    tcldir = os.path.join(sys._MEIPASS, 'tcl')
    tkdir = os.path.join(sys._MEIPASS, 'tk')
    is_darwin = sys.platform == 'darwin'
    if os.path.isdir(tcldir):
        os.environ['TCL_LIBRARY'] = tcldir
    elif not is_darwin:
        raise FileNotFoundError('Tcl data directory "%s" not found.' % tcldir)
    if os.path.isdir(tkdir):
        os.environ['TK_LIBRARY'] = tkdir
    elif not is_darwin:
        raise FileNotFoundError('Tk data directory "%s" not found.' % tkdir)
_pyi_rthook()
del _pyi_rthook