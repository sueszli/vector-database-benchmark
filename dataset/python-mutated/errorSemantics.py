import pythoncom
import winerror
from win32com.client import Dispatch
from win32com.server.exception import COMException
from win32com.server.util import wrap
from win32com.test.util import CaptureWriter

class error(Exception):

    def __init__(self, msg, com_exception=None):
        if False:
            return 10
        Exception.__init__(self, msg, str(com_exception))

class TestServer:
    _public_methods_ = ['Clone', 'Commit', 'LockRegion', 'Read']
    _com_interfaces_ = [pythoncom.IID_IStream]

    def Clone(self):
        if False:
            i = 10
            return i + 15
        raise COMException('Not today', scode=winerror.E_UNEXPECTED)

    def Commit(self, flags):
        if False:
            i = 10
            return i + 15
        if flags == 0:
            raise Exception('😀')
        excepinfo = (winerror.E_UNEXPECTED, 'source', '😀', 'helpfile', 1, winerror.E_FAIL)
        raise pythoncom.com_error(winerror.E_UNEXPECTED, 'desc', excepinfo, None)

def test():
    if False:
        return 10
    com_server = wrap(TestServer(), pythoncom.IID_IStream)
    try:
        com_server.Clone()
        raise error('Expecting this call to fail!')
    except pythoncom.com_error as com_exc:
        if com_exc.hresult != winerror.E_UNEXPECTED:
            raise error('Calling the object natively did not yield the correct scode', com_exc)
        exc = com_exc.excepinfo
        if not exc or exc[-1] != winerror.E_UNEXPECTED:
            raise error('The scode element of the exception tuple did not yield the correct scode', com_exc)
        if exc[2] != 'Not today':
            raise error('The description in the exception tuple did not yield the correct string', com_exc)
    cap = CaptureWriter()
    try:
        cap.capture()
        try:
            com_server.Commit(0)
        finally:
            cap.release()
        raise error('Expecting this call to fail!')
    except pythoncom.com_error as com_exc:
        if com_exc.hresult != winerror.E_FAIL:
            raise error('The hresult was not E_FAIL for an internal error', com_exc)
        if com_exc.excepinfo[1] != 'Python COM Server Internal Error':
            raise error('The description in the exception tuple did not yield the correct string', com_exc)
    if cap.get_captured().find('Traceback') < 0:
        raise error(f'Could not find a traceback in stderr: {cap.get_captured()!r}')
    com_server = Dispatch(wrap(TestServer()))
    try:
        com_server.Clone()
        raise error('Expecting this call to fail!')
    except pythoncom.com_error as com_exc:
        if com_exc.hresult != winerror.DISP_E_EXCEPTION:
            raise error('Calling the object via IDispatch did not yield the correct scode', com_exc)
        exc = com_exc.excepinfo
        if not exc or exc[-1] != winerror.E_UNEXPECTED:
            raise error('The scode element of the exception tuple did not yield the correct scode', com_exc)
        if exc[2] != 'Not today':
            raise error('The description in the exception tuple did not yield the correct string', com_exc)
    cap.clear()
    try:
        cap.capture()
        try:
            com_server.Commit(0)
        finally:
            cap.release()
        raise error('Expecting this call to fail!')
    except pythoncom.com_error as com_exc:
        if com_exc.hresult != winerror.DISP_E_EXCEPTION:
            raise error('Calling the object via IDispatch did not yield the correct scode', com_exc)
        exc = com_exc.excepinfo
        if not exc or exc[-1] != winerror.E_FAIL:
            raise error('The scode element of the exception tuple did not yield the correct scode', com_exc)
        if exc[1] != 'Python COM Server Internal Error':
            raise error('The description in the exception tuple did not yield the correct string', com_exc)
    if cap.get_captured().find('Traceback') < 0:
        raise error(f'Could not find a traceback in stderr: {cap.get_captured()!r}')
    cap.clear()
    try:
        cap.capture()
        try:
            com_server.Commit(1)
        finally:
            cap.release()
        raise error('Expecting this call to fail!')
    except pythoncom.com_error as com_exc:
        if com_exc.hresult != winerror.DISP_E_EXCEPTION:
            raise error('Calling the object via IDispatch did not yield the correct scode', com_exc)
        exc = com_exc.excepinfo
        if not exc or exc[-1] != winerror.E_FAIL:
            raise error('The scode element of the exception tuple did not yield the correct scode', com_exc)
        if exc[1] != 'source':
            raise error('The source in the exception tuple did not yield the correct string', com_exc)
        if exc[2] != '😀':
            raise error('The description in the exception tuple did not yield the correct string', com_exc)
        if exc[3] != 'helpfile':
            raise error('The helpfile in the exception tuple did not yield the correct string', com_exc)
        if exc[4] != 1:
            raise error('The help context in the exception tuple did not yield the correct string', com_exc)
try:
    import logging
except ImportError:
    logging = None
if logging is not None:
    import win32com

    class TestLogHandler(logging.Handler):

        def __init__(self):
            if False:
                return 10
            self.reset()
            logging.Handler.__init__(self)

        def reset(self):
            if False:
                i = 10
                return i + 15
            self.num_emits = 0
            self.last_record = None

        def emit(self, record):
            if False:
                while True:
                    i = 10
            self.num_emits += 1
            self.last_record = self.format(record)
            return
            print('--- record start')
            print(self.last_record)
            print('--- record end')

    def testLogger():
        if False:
            i = 10
            return i + 15
        assert not hasattr(win32com, 'logger')
        handler = TestLogHandler()
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        log = logging.getLogger('win32com_test')
        log.addHandler(handler)
        win32com.logger = log
        com_server = wrap(TestServer(), pythoncom.IID_IStream)
        try:
            com_server.Commit(0)
            raise RuntimeError('should have failed')
        except pythoncom.error as exc:
            message = exc.excepinfo[2]
            assert message.endswith('Exception: 😀\n')
        assert handler.num_emits == 1, handler.num_emits
        assert handler.last_record.startswith("pythoncom error: Unexpected exception in gateway method 'Commit'")
        handler.reset()
        com_server = Dispatch(wrap(TestServer()))
        try:
            com_server.Commit(0)
            raise RuntimeError('should have failed')
        except pythoncom.error as exc:
            message = exc.excepinfo[2]
            assert message.endswith('Exception: 😀\n')
        assert handler.num_emits == 1, handler.num_emits
        handler.reset()
if __name__ == '__main__':
    test()
    if logging is not None:
        testLogger()
    from win32com.test.util import CheckClean
    CheckClean()
    print('error semantic tests worked')