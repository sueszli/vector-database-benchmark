import sys
sys.coinit_flags = 2
import pythoncom
import win32api
from win32com.server import factory
usage = 'Invalid command line arguments\n\nThis program provides LocalServer COM support\nfor Python COM objects.\n\nIt is typically run automatically by COM, passing as arguments\nThe ProgID or CLSID of the Python Server(s) to be hosted\n'

def serve(clsids):
    if False:
        return 10
    infos = factory.RegisterClassFactories(clsids)
    pythoncom.EnableQuitMessage(win32api.GetCurrentThreadId())
    pythoncom.CoResumeClassObjects()
    pythoncom.PumpMessages()
    factory.RevokeClassFactories(infos)
    pythoncom.CoUninitialize()

def main():
    if False:
        i = 10
        return i + 15
    if len(sys.argv) == 1:
        win32api.MessageBox(0, usage, 'Python COM Server')
        sys.exit(1)
    serve(sys.argv[1:])
if __name__ == '__main__':
    main()