import sys
sys.path.insert(1, '../../../')
from tests import pyunit_utils
import h2o

def h2olog_and_echo():
    if False:
        return 10
    "\n    Python API test: h2o.log_and_echo(message=u'')\n    "
    ret = h2o.log_and_echo('Testing h2o.log_and_echo')
    assert ret is None
if __name__ == '__main__':
    pyunit_utils.standalone_test(h2olog_and_echo)
else:
    h2olog_and_echo()