import sys
sys.path.insert(1, '../../../')
from tests import pyunit_utils
import h2o

def h2ocluster_status():
    if False:
        for i in range(10):
            print('nop')
    '\n    Python API test: h2o.cluster_status()\n    Deprecated, use h2o.cluster().show_status(True)\n    '
    ret = h2o.cluster_status()
    assert ret is None
if __name__ == '__main__':
    pyunit_utils.standalone_test(h2ocluster_status)
else:
    h2ocluster_status()