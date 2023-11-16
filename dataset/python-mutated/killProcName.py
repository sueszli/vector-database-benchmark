import sys
import win32api
import win32con
import win32pdhutil

def killProcName(procname):
    if False:
        for i in range(10):
            print('nop')
    try:
        win32pdhutil.GetPerformanceAttributes('Process', 'ID Process', procname)
    except:
        pass
    pids = win32pdhutil.FindPerformanceAttributesByName(procname)
    try:
        pids.remove(win32api.GetCurrentProcessId())
    except ValueError:
        pass
    if len(pids) == 0:
        result = "Can't find %s" % procname
    elif len(pids) > 1:
        result = f"Found too many {procname}'s - pids=`{pids}`"
    else:
        handle = win32api.OpenProcess(win32con.PROCESS_TERMINATE, 0, pids[0])
        win32api.TerminateProcess(handle, 0)
        win32api.CloseHandle(handle)
        result = ''
    return result
if __name__ == '__main__':
    if len(sys.argv) > 1:
        for procname in sys.argv[1:]:
            result = killProcName(procname)
            if result:
                print(result)
                print('Dumping all processes...')
                win32pdhutil.ShowAllProcesses()
            else:
                print('Killed %s' % procname)
    else:
        print('Usage: killProcName.py procname ...')