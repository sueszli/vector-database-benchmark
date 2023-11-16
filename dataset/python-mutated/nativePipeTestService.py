import os
import sys
import servicemanager
import win32serviceutil
from pipeTestService import TestPipeService

class NativeTestPipeService(TestPipeService):
    _svc_name_ = 'PyNativePipeTestService'
    _svc_display_name_ = 'Python Native Pipe Test Service'
    _svc_description_ = 'Tests Python.exe hosted services'
    _exe_name_ = sys.executable
    _exe_args_ = '"' + os.path.abspath(sys.argv[0]) + '"'

def main():
    if False:
        while True:
            i = 10
    if len(sys.argv) == 1:
        print('service is starting...')
        print("(execute this script with '--help' if that isn't what you want)")
        import win32traceutil
        print('service is still starting...')
        servicemanager.Initialize()
        servicemanager.PrepareToHostSingle(NativeTestPipeService)
        servicemanager.StartServiceCtrlDispatcher()
        print('service done!')
    else:
        win32serviceutil.HandleCommandLine(NativeTestPipeService)
if __name__ == '__main__':
    try:
        main()
    except (SystemExit, KeyboardInterrupt):
        raise
    except:
        print('Something went bad!')
        import traceback
        traceback.print_exc()