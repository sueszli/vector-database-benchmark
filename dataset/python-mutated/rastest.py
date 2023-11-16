import os
import sys
import win32ras
stateMap = {}
for (name, val) in list(win32ras.__dict__.items()):
    if name[:6] == 'RASCS_':
        stateMap[val] = name[6:]
import win32event
callbackEvent = win32event.CreateEvent(None, 0, 0, None)

def Callback(hras, msg, state, error, exterror):
    if False:
        while True:
            i = 10
    stateName = stateMap.get(state, 'Unknown state?')
    print('Status is %s (%04lx), error code is %d' % (stateName, state, error))
    finished = state in [win32ras.RASCS_Connected]
    if finished:
        win32event.SetEvent(callbackEvent)
    if error != 0 or int(state) == win32ras.RASCS_Disconnected:
        print('Detected call failure: %s' % win32ras.GetErrorString(error))
        HangUp(hras)
        win32event.SetEvent(callbackEvent)

def ShowConnections():
    if False:
        for i in range(10):
            print('nop')
    print('All phone-book entries:')
    for (name,) in win32ras.EnumEntries():
        print(' ', name)
    print('Current Connections:')
    for con in win32ras.EnumConnections():
        print(' ', con)

def EditEntry(entryName):
    if False:
        for i in range(10):
            print('nop')
    try:
        win32ras.EditPhonebookEntry(0, None, entryName)
    except win32ras.error as xxx_todo_changeme:
        (rc, function, msg) = xxx_todo_changeme.args
        print('Can not edit/find the RAS entry -', msg)

def HangUp(hras):
    if False:
        i = 10
        return i + 15
    try:
        win32ras.HangUp(hras)
    except:
        print("Tried to hang up gracefully on error, but didn't work....")
    return None

def Connect(entryName, bUseCallback):
    if False:
        i = 10
        return i + 15
    if bUseCallback:
        theCallback = Callback
        win32event.ResetEvent(callbackEvent)
    else:
        theCallback = None
    try:
        (dp, b) = win32ras.GetEntryDialParams(None, entryName)
    except:
        print("Couldn't find DUN entry: %s" % entryName)
    else:
        (hras, rc) = win32ras.Dial(None, None, (entryName, '', '', dp[3], dp[4], ''), theCallback)
        if not bUseCallback and rc != 0:
            print('Could not dial the RAS connection:', win32ras.GetErrorString(rc))
            hras = HangUp(hras)
        elif bUseCallback and win32event.WaitForSingleObject(callbackEvent, 60000) != win32event.WAIT_OBJECT_0:
            print('Gave up waiting for the process to complete!')
            try:
                cs = win32ras.GetConnectStatus(hras)
            except:
                hras = HangUp(hras)
            else:
                if int(cs[0]) == win32ras.RASCS_Disconnected:
                    hras = HangUp(hras)
    return (hras, rc)

def Disconnect(rasEntry):
    if False:
        print('Hello World!')
    name = rasEntry.lower()
    for (hcon, entryName, devName, devType) in win32ras.EnumConnections():
        if entryName.lower() == name:
            win32ras.HangUp(hcon)
            print('Disconnected from', rasEntry)
            break
    else:
        print('Could not find an open connection to', entryName)
usage = '\nUsage: %s [-s] [-l] [-c connection] [-d connection]\n-l : List phone-book entries and current connections.\n-s : Show status while connecting/disconnecting (uses callbacks)\n-c : Connect to the specified phonebook name.\n-d : Disconnect from the specified phonebook name.\n-e : Edit the specified phonebook entry.\n'

def main():
    if False:
        while True:
            i = 10
    import getopt
    try:
        (opts, args) = getopt.getopt(sys.argv[1:], 'slc:d:e:')
    except getopt.error as why:
        print(why)
        print(usage % os.path.basename(sys.argv[0]))
        return
    bCallback = 0
    if args or not opts:
        print(usage % os.path.basename(sys.argv[0]))
        return
    for (opt, val) in opts:
        if opt == '-s':
            bCallback = 1
        if opt == '-l':
            ShowConnections()
        if opt == '-c':
            (hras, rc) = Connect(val, bCallback)
            if hras is not None:
                print(f'hras: 0x{hras:8x}, rc: 0x{rc:04x}')
        if opt == '-d':
            Disconnect(val)
        if opt == '-e':
            EditEntry(val)
if __name__ == '__main__':
    main()