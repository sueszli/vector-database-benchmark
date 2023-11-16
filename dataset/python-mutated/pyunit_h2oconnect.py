import sys
sys.path.insert(1, '../../../')
from tests import pyunit_utils
import h2o
from h2o.utils.typechecks import assert_is_type
from h2o.backend.connection import H2OConnection

def h2oconnect():
    if False:
        while True:
            i = 10
    '\n    Python API test: h2o.connect(server=None, url=None, ip=None, port=None, https=None, verify_ssl_certificates=None,\n     auth=None, proxy=None,cookies=None, verbose=True)\n    '
    ipA = '127.0.0.1'
    portN = '54321'
    urlS = 'http://127.0.0.1:54321'
    try:
        connect_type = h2o.connect(ip=ipA, port=portN, verbose=True)
        assert_is_type(connect_type, H2OConnection)
    except Exception as e:
        assert 'Could not establish link' in e.args[0], 'h2o.connect command is not working.'
    try:
        connect_type2 = h2o.connect(url=urlS, https=True, verbose=True)
        assert_is_type(connect_type2, H2OConnection)
    except Exception as e:
        assert 'Could not establish link' in e.args[0], 'h2o.connect command is not working.'
if __name__ == '__main__':
    pyunit_utils.standalone_test(h2oconnect)
else:
    h2oconnect()