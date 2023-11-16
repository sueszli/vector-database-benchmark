import sys
sys.path.insert(1, '../../../')
import h2o
from tests import pyunit_utils
from h2o.utils.typechecks import assert_is_type
from h2o.frame import H2OFrame

def h2o_H2OFrame_na_omit():
    if False:
        return 10
    '\n    Python API test: h2o.frame.H2OFrame.na_omit()\n\n    Copied from runit_lstrip.R\n    '
    iris = h2o.import_file(path=pyunit_utils.locate('smalldata/iris/iris_wheader_NA_2.csv'))
    newframe = iris.na_omit()
    assert_is_type(newframe, H2OFrame)
    assert newframe.nrow == iris.nrow - 10, 'h2o.H2OFrame.na_omit() command is not working.'
pyunit_utils.standalone_test(h2o_H2OFrame_na_omit)