import getopt
import os
import sys
import win32api
import win32con
import win32event
import wincerapi

def DumpPythonRegistry():
    if False:
        i = 10
        return i + 15
    try:
        h = wincerapi.CeRegOpenKeyEx(win32con.HKEY_LOCAL_MACHINE, 'Software\\Python\\PythonCore\\%s\\PythonPath' % sys.winver)
    except win32api.error:
        print('The remote device does not appear to have Python installed')
        return 0
    (path, typ) = wincerapi.CeRegQueryValueEx(h, None)
    print(f"The remote PythonPath is '{str(path)}'")
    h.Close()
    return 1

def DumpRegistry(root, level=0):
    if False:
        for i in range(10):
            print('nop')
    h = wincerapi.CeRegOpenKeyEx(win32con.HKEY_LOCAL_MACHINE, None)
    level_prefix = ' ' * level
    index = 0
    while 1:
        try:
            (name, data, typ) = wincerapi.CeRegEnumValue(root, index)
        except win32api.error:
            break
        print(f'{level_prefix}{name}={repr(str(data))}')
        index = index + 1
    index = 0
    while 1:
        try:
            (name, klass) = wincerapi.CeRegEnumKeyEx(root, index)
        except win32api.error:
            break
        print(f'{level_prefix}{name}\\')
        subkey = wincerapi.CeRegOpenKeyEx(root, name)
        DumpRegistry(subkey, level + 1)
        index = index + 1

def DemoCopyFile():
    if False:
        return 10
    cefile = wincerapi.CeCreateFile('TestPython', win32con.GENERIC_WRITE, 0, None, win32con.OPEN_ALWAYS, 0, None)
    wincerapi.CeWriteFile(cefile, 'Hello from Python')
    cefile.Close()
    cefile = wincerapi.CeCreateFile('TestPython', win32con.GENERIC_READ, 0, None, win32con.OPEN_EXISTING, 0, None)
    if wincerapi.CeReadFile(cefile, 100) != 'Hello from Python':
        print('Couldnt read the data from the device!')
    cefile.Close()
    wincerapi.CeDeleteFile('TestPython')
    print('Created, wrote to, read from and deleted a test file!')

def DemoCreateProcess():
    if False:
        while True:
            i = 10
    try:
        (hp, ht, pid, tid) = wincerapi.CeCreateProcess('Windows\\Python.exe', '', None, None, 0, 0, None, '', None)
        hp.Close()
        ht.Close()
        print('Python is running on the remote device!')
    except win32api.error as xxx_todo_changeme1:
        (hr, fn, msg) = xxx_todo_changeme1.args
        print('Couldnt execute remote process -', msg)

def DumpRemoteMachineStatus():
    if False:
        while True:
            i = 10
    (ACLineStatus, BatteryFlag, BatteryLifePercent, BatteryLifeTime, BatteryFullLifeTime, BackupBatteryFlag, BackupBatteryLifePercent, BackupBatteryLifeTime, BackupBatteryLifeTime) = wincerapi.CeGetSystemPowerStatusEx()
    if ACLineStatus:
        power = 'AC'
    else:
        power = 'battery'
    if BatteryLifePercent == 255:
        batPerc = 'unknown'
    else:
        batPerc = BatteryLifePercent
    print(f'The batteries are at {batPerc}%, and is currently being powered by {power}')
    (memLoad, totalPhys, availPhys, totalPage, availPage, totalVirt, availVirt) = wincerapi.CeGlobalMemoryStatus()
    print('The memory is %d%% utilized.' % memLoad)
    print('%-20s%-10s%-10s' % ('', 'Total', 'Avail'))
    print('%-20s%-10s%-10s' % ('Physical Memory', totalPhys, availPhys))
    print('%-20s%-10s%-10s' % ('Virtual Memory', totalVirt, availVirt))
    print('%-20s%-10s%-10s' % ('Paging file', totalPage, availPage))
    (storeSize, freeSize) = wincerapi.CeGetStoreInformation()
    print('%-20s%-10s%-10s' % ('File store', storeSize, freeSize))
    print('The CE temp path is', wincerapi.CeGetTempPath())
    print('The system info for the device is', wincerapi.CeGetSystemInfo())

def DumpRemoteFolders():
    if False:
        i = 10
        return i + 15
    for (name, val) in list(wincerapi.__dict__.items()):
        if name[:6] == 'CSIDL_':
            try:
                loc = str(wincerapi.CeGetSpecialFolderPath(val))
                print(f'Folder {name} is at {loc}')
            except win32api.error as details:
                pass
    print('Dumping start menu shortcuts...')
    try:
        startMenu = str(wincerapi.CeGetSpecialFolderPath(wincerapi.CSIDL_STARTMENU))
    except win32api.error as details:
        print('This device has no start menu!', details)
        startMenu = None
    if startMenu:
        for fileAttr in wincerapi.CeFindFiles(os.path.join(startMenu, '*')):
            fileName = fileAttr[8]
            fullPath = os.path.join(startMenu, str(fileName))
            try:
                resolved = wincerapi.CeSHGetShortcutTarget(fullPath)
            except win32api.error as xxx_todo_changeme:
                (rc, fn, msg) = xxx_todo_changeme.args
                resolved = '#Error - %s' % msg
            print(f'{fileName}->{resolved}')

def usage():
    if False:
        while True:
            i = 10
    print('Options:')
    print('-a - Execute all demos')
    print('-p - Execute Python process on remote device')
    print('-r - Dump the remote registry')
    print('-f - Dump all remote special folder locations')
    print('-s - Dont dump machine status')
    print('-y - Perform asynch init of CE connection')

def main():
    if False:
        for i in range(10):
            print('nop')
    async_init = bStartPython = bDumpRegistry = bDumpFolders = 0
    bDumpStatus = 1
    try:
        (opts, args) = getopt.getopt(sys.argv[1:], 'apr')
    except getopt.error as why:
        print('Invalid usage:', why)
        usage()
        return
    for (o, v) in opts:
        if o == '-a':
            bStartPython = bDumpRegistry = bDumpStatus = bDumpFolders = asynch_init = 1
        if o == '-p':
            bStartPython = 1
        if o == '-r':
            bDumpRegistry = 1
        if o == '-s':
            bDumpStatus = 0
        if o == '-f':
            bDumpFolders = 1
        if o == '-y':
            print('Doing asynch init of CE connection')
            async_init = 1
    if async_init:
        (event, rc) = wincerapi.CeRapiInitEx()
        while 1:
            rc = win32event.WaitForSingleObject(event, 500)
            if rc == win32event.WAIT_OBJECT_0:
                break
            else:
                print('Waiting for Initialize to complete (picture a Cancel button here :)')
    else:
        wincerapi.CeRapiInit()
    print('Connected to remote CE device.')
    try:
        verinfo = wincerapi.CeGetVersionEx()
        print('The device is running windows CE version %d.%d - %s' % (verinfo[0], verinfo[1], verinfo[4]))
        if bDumpStatus:
            print('Dumping remote machine status')
            DumpRemoteMachineStatus()
        if bDumpRegistry:
            print('Dumping remote registry...')
            DumpRegistry(win32con.HKEY_LOCAL_MACHINE)
        if bDumpFolders:
            print('Dumping remote folder information')
            DumpRemoteFolders()
        DemoCopyFile()
        if bStartPython:
            print('Starting remote Python process')
            if DumpPythonRegistry():
                DemoCreateProcess()
            else:
                print("Not trying to start Python, as it's not installed")
    finally:
        wincerapi.CeRapiUninit()
        print('Disconnected')
if __name__ == '__main__':
    main()