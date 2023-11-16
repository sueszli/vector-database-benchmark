import sys
import urllib.error
import urllib.parse
import urllib.request
from isapi import isapicon, threaded_extension
if hasattr(sys, 'isapidllhandle'):
    import win32traceutil
proxy = 'http://www.python.org'
CHUNK_SIZE = 8192

def io_callback(ecb, fp, cbIO, errcode):
    if False:
        while True:
            i = 10
    print('IO callback', ecb, fp, cbIO, errcode)
    chunk = fp.read(CHUNK_SIZE)
    if chunk:
        ecb.WriteClient(chunk, isapicon.HSE_IO_ASYNC)
    else:
        fp.close()
        ecb.DoneWithSession()

class Extension(threaded_extension.ThreadPoolExtension):
    """Python sample proxy server - asynch version."""

    def Dispatch(self, ecb):
        if False:
            while True:
                i = 10
        print('IIS dispatching "{}"'.format(ecb.GetServerVariable('URL')))
        url = ecb.GetServerVariable('URL')
        new_url = proxy + url
        print('Opening %s' % new_url)
        fp = urllib.request.urlopen(new_url)
        headers = fp.info()
        ecb.SendResponseHeaders('200 OK', str(headers) + '\r\n', False)
        ecb.ReqIOCompletion(io_callback, fp)
        chunk = fp.read(CHUNK_SIZE)
        if chunk:
            ecb.WriteClient(chunk, isapicon.HSE_IO_ASYNC)
            return isapicon.HSE_STATUS_PENDING
        ecb.DoneWithSession()
        return isapicon.HSE_STATUS_SUCCESS

def __ExtensionFactory__():
    if False:
        while True:
            i = 10
    return Extension()
if __name__ == '__main__':
    from isapi.install import *
    params = ISAPIParameters()
    sm = [ScriptMapParams(Extension='*', Flags=0)]
    vd = VirtualDirParameters(Name='/', Description=Extension.__doc__, ScriptMaps=sm, ScriptMapUpdate='replace')
    params.VirtualDirs = [vd]
    HandleCommandLine(params)