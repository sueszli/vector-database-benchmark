import win32api
import winerror
from isapi import ExtensionError, threaded_extension
try:
    win32api.GetConsoleTitle()
except win32api.error:
    import win32traceutil

class Extension(threaded_extension.ThreadPoolExtension):
    """Python ISAPI Tester"""

    def Dispatch(self, ecb):
        if False:
            while True:
                i = 10
        print('Tester dispatching "{}"'.format(ecb.GetServerVariable('URL')))
        url = ecb.GetServerVariable('URL')
        test_name = url.split('/')[-1]
        meth = getattr(self, test_name, None)
        if meth is None:
            raise AttributeError(f"No test named '{test_name}'")
        result = meth(ecb)
        if result is None:
            return
        ecb.SendResponseHeaders('200 OK', 'Content-type: text/html\r\n\r\n', False)
        print('<HTML><BODY>Finished running test <i>', test_name, '</i>', file=ecb)
        print('<pre>', file=ecb)
        print(result, file=ecb)
        print('</pre>', file=ecb)
        print('</BODY></HTML>', file=ecb)
        ecb.DoneWithSession()

    def test1(self, ecb):
        if False:
            print('Hello World!')
        try:
            ecb.GetServerVariable('foo bar')
            raise RuntimeError('should have failed!')
        except ExtensionError as err:
            assert err.errno == winerror.ERROR_INVALID_INDEX, err
        return 'worked!'

    def test_long_vars(self, ecb):
        if False:
            print('Hello World!')
        qs = ecb.GetServerVariable('QUERY_STRING')
        expected_query = 'x' * 8500
        if len(qs) == 0:
            me = ecb.GetServerVariable('URL')
            headers = 'Location: ' + me + '?' + expected_query + '\r\n\r\n'
            ecb.SendResponseHeaders('301 Moved', headers)
            ecb.DoneWithSession()
            return None
        if qs == expected_query:
            return 'Total length of variable is %d - test worked!' % (len(qs),)
        else:
            return 'Unexpected query portion!  Got %d chars, expected %d' % (len(qs), len(expected_query))

    def test_unicode_vars(self, ecb):
        if False:
            return 10
        ver = float(ecb.GetServerVariable('SERVER_SOFTWARE').split('/')[1])
        if ver < 6.0:
            return 'This is IIS version %g - unicode only works in IIS6 and later' % ver
        us = ecb.GetServerVariable('UNICODE_SERVER_NAME')
        if not isinstance(us, str):
            raise RuntimeError('unexpected type!')
        if us != str(ecb.GetServerVariable('SERVER_NAME')):
            raise RuntimeError('Unicode and non-unicode values were not the same')
        return 'worked!'

def __ExtensionFactory__():
    if False:
        i = 10
        return i + 15
    return Extension()
if __name__ == '__main__':
    from isapi.install import *
    params = ISAPIParameters()
    sm = [ScriptMapParams(Extension='*', Flags=0)]
    vd = VirtualDirParameters(Name='pyisapi_test', Description=Extension.__doc__, ScriptMaps=sm, ScriptMapUpdate='replace')
    params.VirtualDirs = [vd]
    HandleCommandLine(params)