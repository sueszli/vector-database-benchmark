import sys
sys.path.insert(1, '../../../')
from tests import pyunit_utils
import h2o

def h2onetwork_test():
    if False:
        while True:
            i = 10
    '\n    Python API test: h2o.network_test()\n    Deprecated, use h2o.cluster().network_test().\n    '
    ret = h2o.network_test()
    assert ret is None
if __name__ == '__main__':
    pyunit_utils.standalone_test(h2onetwork_test)
else:
    h2onetwork_test()