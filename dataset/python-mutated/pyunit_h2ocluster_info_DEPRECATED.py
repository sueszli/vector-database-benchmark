import sys
sys.path.insert(1, '../../../')
from tests import pyunit_utils
import h2o

def h2ocluster_info():
    if False:
        for i in range(10):
            print('nop')
    '\n    Python API test: h2o.cluster_info()\n    Deprecated, use h2o.cluster().show_status().\n    '
    ret = h2o.cluster_info()
    assert ret is None
if __name__ == '__main__':
    pyunit_utils.standalone_test(h2ocluster_info)
else:
    h2ocluster_info()