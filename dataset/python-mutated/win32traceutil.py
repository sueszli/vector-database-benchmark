import win32trace

def RunAsCollector():
    if False:
        print('Hello World!')
    import sys
    try:
        import win32api
        win32api.SetConsoleTitle('Python Trace Collector')
    except:
        pass
    win32trace.InitRead()
    print('Collecting Python Trace Output...')
    try:
        while 1:
            sys.stdout.write(win32trace.blockingread(500))
    except KeyboardInterrupt:
        print('Ctrl+C')

def SetupForPrint():
    if False:
        while True:
            i = 10
    win32trace.InitWrite()
    try:
        print('Redirecting output to win32trace remote collector')
    except:
        pass
    win32trace.setprint()
if __name__ == '__main__':
    RunAsCollector()
else:
    SetupForPrint()