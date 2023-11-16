import unittest
import sys
from io import StringIO
from threading import Thread
from mycroft.util.log import LOG

class CaptureLogs(list):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        list.__init__(self)
        self._stdout = None
        self._stringio = None

    def __enter__(self):
        if False:
            for i in range(10):
                print('nop')
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        LOG.init()
        return self

    def __exit__(self, *args):
        if False:
            for i in range(10):
                print('nop')
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio
        sys.stdout = self._stdout
        LOG.init()

class TestLog(unittest.TestCase):

    def test_threads(self):
        if False:
            i = 10
            return i + 15
        with CaptureLogs() as output:

            def test_logging():
                if False:
                    for i in range(10):
                        print('nop')
                LOG.debug('testing debug')
                LOG.info('testing info')
                LOG.warning('testing warning')
                LOG.error('testing error')
                LOG('testing custom').debug('test')
            threads = []
            for _ in range(100):
                t = Thread(target=test_logging)
                t.start()
                threads.append(t)
            for t in threads:
                t.join()
        assert len(output) > 0
        for line in output:
            found_msg = False
            for msg in ['debug', 'info', 'warning', 'error', 'custom']:
                if 'testing ' + msg in line:
                    found_msg = True
            assert found_msg
if __name__ == '__main__':
    unittest.main()