import glob
import os
import sys
import tkinter

def compare(test_name, expect, frozen):
    if False:
        return 10
    expect = os.path.normpath(expect)
    print(test_name)
    print('  Expected: ' + expect)
    print('  Current:  ' + frozen)
    print('')
    if not frozen == expect:
        raise SystemExit('Data directory is not set properly.')
    if not os.path.exists(frozen):
        raise SystemExit('Data directory does not exist.')
    if not len(glob.glob(frozen + '/*.tcl')) > 0:
        raise SystemExit('Data directory does not contain .tcl files.')
tcl_dir = os.environ.get('TCL_LIBRARY')
if tcl_dir:
    compare('Tcl', os.path.join(sys.prefix, 'tcl'), tcl_dir)
elif sys.platform != 'darwin':
    raise SystemExit('TCL_LIBRARY environment variable is not set!')
tk_dir = os.environ.get('TK_LIBRARY')
if tk_dir:
    compare('Tk', os.path.join(sys.prefix, 'tk'), tk_dir)
elif sys.platform != 'darwin':
    raise SystemExit('TK_LIBRARY environment variable is not set!')