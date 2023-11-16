import sys
import os
import win32api

def check_shortpathname(fn):
    if False:
        while True:
            i = 10
    lfn = win32api.GetLongPathNameW(fn)
    fn = os.path.normcase(fn)
    lfn = os.path.normcase(lfn)
    if lfn != fn:
        print('ShortPathName: Expected %s, got %s' % (fn, lfn))
        raise SystemExit(-1)
print('sys.executable:', ascii(sys.executable))
if not os.path.exists(sys.executable):
    raise SystemExit('sys.executable does not exist.')
check_shortpathname(sys.executable)
print('sys.argv[0]:', ascii(sys.argv[0]))
if not os.path.exists(sys.argv[0]):
    raise SystemExit('sys.argv[0] does not exist.')
check_shortpathname(sys.argv[0])
print('sys._MEIPASS:', ascii(sys._MEIPASS))
if not os.path.exists(sys._MEIPASS):
    raise SystemExit('sys._MEIPASS does not exist.')
tmp = os.path.normcase(win32api.GetTempPath())
if os.path.normcase(win32api.GetLongPathNameW(tmp)) == tmp:
    check_shortpathname(sys._MEIPASS)