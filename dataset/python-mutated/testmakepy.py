import sys
import traceback
import pythoncom
import win32api
import win32com.test.util
import winerror
from win32com.client import gencache, makepy, selecttlb

def TestBuildAll(verbose=1):
    if False:
        while True:
            i = 10
    num = 0
    tlbInfos = selecttlb.EnumTlbs()
    for info in tlbInfos:
        if verbose:
            print(f'{info.desc} ({info.dll})')
        try:
            makepy.GenerateFromTypeLibSpec(info)
            num += 1
        except pythoncom.com_error as details:
            if details.hresult not in [winerror.TYPE_E_CANTLOADLIBRARY, winerror.TYPE_E_LIBNOTREGISTERED]:
                print('** COM error on', info.desc)
                print(details)
        except KeyboardInterrupt:
            print('Interrupted!')
            raise KeyboardInterrupt
        except:
            print('Failed:', info.desc)
            traceback.print_exc()
        if makepy.bForDemandDefault:
            tinfo = (info.clsid, info.lcid, info.major, info.minor)
            mod = gencache.EnsureModule(info.clsid, info.lcid, info.major, info.minor)
            for name in mod.NamesToIIDMap.keys():
                makepy.GenerateChildFromTypeLibSpec(name, tinfo)
    return num

def TestAll(verbose=0):
    if False:
        print('Hello World!')
    num = TestBuildAll(verbose)
    print('Generated and imported', num, 'modules')
    win32com.test.util.CheckClean()
if __name__ == '__main__':
    TestAll('-q' not in sys.argv)