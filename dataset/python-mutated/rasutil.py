import sys
import win32ras

class ConnectionError(Exception):
    pass

def Connect(rasEntryName, numRetries=5):
    if False:
        for i in range(10):
            print('nop')
    'Make a connection to the specified RAS entry.\n\n    Returns a tuple of (bool, handle) on success.\n    - bool is 1 if a new connection was established, or 0 is a connection already existed.\n    - handle is a RAS HANDLE that can be passed to Disconnect() to end the connection.\n\n    Raises a ConnectionError if the connection could not be established.\n    '
    assert numRetries > 0
    for info in win32ras.EnumConnections():
        if info[1].lower() == rasEntryName.lower():
            print('Already connected to', rasEntryName)
            return (0, info[0])
    (dial_params, have_pw) = win32ras.GetEntryDialParams(None, rasEntryName)
    if not have_pw:
        print('Error: The password is not saved for this connection')
        print("Please connect manually selecting the 'save password' option and try again")
        sys.exit(1)
    print('Connecting to', rasEntryName, '...')
    retryCount = numRetries
    while retryCount > 0:
        (rasHandle, errCode) = win32ras.Dial(None, None, dial_params, None)
        if win32ras.IsHandleValid(rasHandle):
            bValid = 1
            break
        print('Retrying...')
        win32api.Sleep(5000)
        retryCount = retryCount - 1
    if errCode:
        raise ConnectionError(errCode, win32ras.GetErrorString(errCode))
    return (1, rasHandle)

def Disconnect(handle):
    if False:
        for i in range(10):
            print('nop')
    if isinstance(handle, str):
        for info in win32ras.EnumConnections():
            if info[1].lower() == handle.lower():
                handle = info[0]
                break
        else:
            raise ConnectionError(0, "Not connected to entry '%s'" % handle)
    win32ras.HangUp(handle)
usage = 'rasutil.py - Utilities for using RAS\n\nUsage:\n  rasutil [-r retryCount] [-c rasname] [-d rasname]\n  \n  -r retryCount - Number of times to retry the RAS connection\n  -c rasname - Connect to the phonebook entry specified by rasname\n  -d rasname - Disconnect from the phonebook entry specified by rasname\n'

def Usage(why):
    if False:
        while True:
            i = 10
    print(why)
    print(usage)
    sys.exit(1)
if __name__ == '__main__':
    import getopt
    try:
        (opts, args) = getopt.getopt(sys.argv[1:], 'r:c:d:')
    except getopt.error as why:
        Usage(why)
    retries = 5
    if len(args) != 0:
        Usage('Invalid argument')
    for (opt, val) in opts:
        if opt == '-c':
            Connect(val, retries)
        if opt == '-d':
            Disconnect(val)
        if opt == '-r':
            retries = int(val)