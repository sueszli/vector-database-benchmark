import sys
import os
sys.path.insert(1, os.path.join('../../../h2o-py'))
from tests import pyunit_utils
import h2o
from h2o.exceptions import H2OServerError

def trace_request():
    if False:
        for i in range(10):
            print('nop')
    err = None
    try:
        h2o.api('TRACE /3/Cloud')
    except H2OServerError as e:
        err = e
    msg = str(err.args[0])
    assert err is not None
    print('<Error message>')
    print(msg)
    print('</Error Message>')
    assert msg.startswith('HTTP 500') or msg.startswith('HTTP 405 Method Not Allowed')
if __name__ == '__main__':
    pyunit_utils.standalone_test(trace_request)
else:
    trace_request()