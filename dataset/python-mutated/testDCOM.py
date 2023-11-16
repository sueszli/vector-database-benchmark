usage = 'testDCOM.py - Simple DCOM test\nUsage: testDCOM.py serverName\n\nAttempts to start the Python.Interpreter object on the named machine,\nand checks that the object is indeed running remotely.\n\nRequires the named server be configured to run DCOM (using dcomcnfg.exe),\nand the Python.Interpreter object installed and registered on that machine.\n\nThe Python.Interpreter object must be installed on the local machine,\nbut no special DCOM configuration should be necessary.\n'
import sys
import pythoncom
import win32api
import win32com.client

def test(serverName):
    if False:
        i = 10
        return i + 15
    if serverName.lower() == win32api.GetComputerName().lower():
        print('You must specify a remote server name, not the local machine!')
        return
    clsctx = pythoncom.CLSCTX_SERVER & ~pythoncom.CLSCTX_INPROC_SERVER
    ob = win32com.client.DispatchEx('Python.Interpreter', serverName, clsctx=clsctx)
    ob.Exec('import win32api')
    actualName = ob.Eval('win32api.GetComputerName()')
    if serverName.lower() != actualName.lower():
        print("Error: The object created on server '{}' reported its name as '{}'".format(serverName, actualName))
    else:
        print("Object created and tested OK on server '%s'" % serverName)
if __name__ == '__main__':
    if len(sys.argv) == 2:
        test(sys.argv[1])
    else:
        print(usage)