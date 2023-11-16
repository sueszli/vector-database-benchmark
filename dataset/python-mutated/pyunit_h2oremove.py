import sys
sys.path.insert(1, '../../../')
from tests import pyunit_utils
import h2o
from h2o.utils.typechecks import assert_is_type
from h2o.frame import H2OFrame

def h2oremove():
    if False:
        while True:
            i = 10
    '\n    Python API test: h2o.remove(x)\n    '
    try:
        training_data = h2o.import_file(pyunit_utils.locate('smalldata/logreg/benign.csv'))
        assert_is_type(training_data, H2OFrame)
        h2o.remove(training_data)
        training_data.nrow
    except Exception as e:
        assert_is_type(e, AttributeError)
        assert 'object has no attribute' in e.args[0], 'h2o.remove() command is not working.'
if __name__ == '__main__':
    pyunit_utils.standalone_test(h2oremove)
else:
    h2oremove()