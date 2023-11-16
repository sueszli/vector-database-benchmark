import sys
from urllib.request import urlopen
import win32api
from isapi import isapicon, threaded_extension
if hasattr(sys, 'isapidllhandle'):
    import win32traceutil
proxy = 'http://www.python.org'
excludes = ['/iisstart.htm', '/welcome.png']

def io_callback(ecb, url, cbIO, errcode):
    if False:
        for i in range(10):
            print('nop')
    (httpstatus, substatus, win32) = ecb.GetExecURLStatus()
    print('ExecURL of %r finished with http status %d.%d, win32 status %d (%s)' % (url, httpstatus, substatus, win32, win32api.FormatMessage(win32).strip()))
    ecb.DoneWithSession()

class Extension(threaded_extension.ThreadPoolExtension):
    """Python sample Extension"""

    def Dispatch(self, ecb):
        if False:
            for i in range(10):
                print('nop')
        url = ecb.GetServerVariable('URL').decode('ascii')
        for exclude in excludes:
            if url.lower().startswith(exclude):
                print('excluding %s' % url)
                if ecb.Version < 393216:
                    print("(but this is IIS5 or earlier - can't do 'excludes')")
                else:
                    ecb.IOCompletion(io_callback, url)
                    ecb.ExecURL(None, None, None, None, None, isapicon.HSE_EXEC_URL_IGNORE_CURRENT_INTERCEPTOR)
                    return isapicon.HSE_STATUS_PENDING
        new_url = proxy + url
        print('Opening %s' % new_url)
        fp = urlopen(new_url)
        headers = fp.info()
        header_text = str(headers).rstrip('\n').replace('\n', '\r\n') + '\r\n\r\n'
        ecb.SendResponseHeaders('200 OK', header_text, False)
        ecb.WriteClient(fp.read())
        ecb.DoneWithSession()
        print(f"Returned data from '{new_url}'")
        return isapicon.HSE_STATUS_SUCCESS

def __ExtensionFactory__():
    if False:
        for i in range(10):
            print('nop')
    return Extension()
if __name__ == '__main__':
    from isapi.install import *
    params = ISAPIParameters()
    sm = [ScriptMapParams(Extension='*', Flags=0)]
    vd = VirtualDirParameters(Name='/', Description=Extension.__doc__, ScriptMaps=sm, ScriptMapUpdate='replace')
    params.VirtualDirs = [vd]
    HandleCommandLine(params)