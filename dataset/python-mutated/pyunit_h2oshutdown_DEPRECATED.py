import sys
sys.path.insert(1, '../../../')
from tests import pyunit_utils
import h2o
import threading
from h2o.utils.typechecks import assert_is_type

def h2oshutdown():
    if False:
        return 10
    '\n    Python API test: h2o.shutdown(prompt=False)\n    Deprecated, use h2o.cluster().shutdown()\n    '
    try:
        bthread = threading.Thread(target=call_badshutdown())
        bthread.daemon = True
        bthread.start()
        bthread.join(1.0)
    except Exception as e:
        print('*** Error in thread is caught=> ')
        print(e)
        assert_is_type(e, TypeError)
        assert 'badparam' in e.args[0], 'h2o.shutdown() command is not working.'
    thread = threading.Thread(target=call_shutdown)
    thread.daemon = True
    thread.start()
    thread.join(1.0)

def call_shutdown():
    if False:
        i = 10
        return i + 15
    h2o.shutdown(prompt=True)

def call_badshutdown():
    if False:
        return 10
    h2o.shutdown(badparam=1, prompt=True)
if __name__ == '__main__':
    pyunit_utils.standalone_test(h2oshutdown)
else:
    h2oshutdown()