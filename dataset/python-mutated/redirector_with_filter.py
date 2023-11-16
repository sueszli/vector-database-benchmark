import sys
import urllib.error
import urllib.parse
import urllib.request
from isapi import isapicon, threaded_extension
from isapi.simple import SimpleFilter
if hasattr(sys, 'isapidllhandle'):
    import win32traceutil
proxy = 'http://www.python.org'
virtualdir = '/python'

class Extension(threaded_extension.ThreadPoolExtension):
    """Python sample Extension"""

    def Dispatch(self, ecb):
        if False:
            for i in range(10):
                print('nop')
        url = ecb.GetServerVariable('URL')
        if url.startswith(virtualdir):
            new_url = proxy + url[len(virtualdir):]
            print('Opening', new_url)
            fp = urllib.request.urlopen(new_url)
            headers = fp.info()
            ecb.SendResponseHeaders('200 OK', str(headers) + '\r\n', False)
            ecb.WriteClient(fp.read())
            ecb.DoneWithSession()
            print(f"Returned data from '{new_url}'!")
        else:
            print(f"Not proxying '{url}'")

class Filter(SimpleFilter):
    """Sample Python Redirector"""
    filter_flags = isapicon.SF_NOTIFY_PREPROC_HEADERS | isapicon.SF_NOTIFY_ORDER_DEFAULT

    def HttpFilterProc(self, fc):
        if False:
            for i in range(10):
                print('nop')
        nt = fc.NotificationType
        if nt != isapicon.SF_NOTIFY_PREPROC_HEADERS:
            return isapicon.SF_STATUS_REQ_NEXT_NOTIFICATION
        pp = fc.GetData()
        url = pp.GetHeader('url')
        prefix = virtualdir
        if not url.startswith(prefix):
            new_url = prefix + url
            print(f"New proxied URL is '{new_url}'")
            pp.SetHeader('url', new_url)
            if fc.FilterContext is None:
                fc.FilterContext = 0
            fc.FilterContext += 1
            print('This is request number %d on this connection' % fc.FilterContext)
            return isapicon.SF_STATUS_REQ_HANDLED_NOTIFICATION
        else:
            print(f"Filter ignoring URL '{url}'")

def __FilterFactory__():
    if False:
        return 10
    return Filter()

def __ExtensionFactory__():
    if False:
        while True:
            i = 10
    return Extension()
if __name__ == '__main__':
    from isapi.install import *
    params = ISAPIParameters()
    params.Filters = [FilterParameters(Name='PythonRedirector', Description=Filter.__doc__)]
    sm = [ScriptMapParams(Extension='*', Flags=0)]
    vd = VirtualDirParameters(Name=virtualdir[1:], Description=Extension.__doc__, ScriptMaps=sm, ScriptMapUpdate='replace')
    params.VirtualDirs = [vd]
    HandleCommandLine(params)