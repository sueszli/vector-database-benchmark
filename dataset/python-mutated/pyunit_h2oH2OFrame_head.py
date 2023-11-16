import sys
sys.path.insert(1, '../../../')
import h2o
from tests import pyunit_utils
from random import randrange
from h2o.frame import H2OFrame
from h2o.utils.typechecks import assert_is_type

def h2o_H2OFrame_head():
    if False:
        while True:
            i = 10
    '\n    Python API test: h2o.frame.H2OFrame.head(rows=10, cols=200)\n    '
    frame = h2o.import_file(path=pyunit_utils.locate('smalldata/iris/iris.csv'), col_types=['numeric', 'numeric', 'numeric', 'numeric', 'string'])
    rowNum = randrange(1, frame.nrow)
    colNum = randrange(1, frame.ncol)
    newFrame = frame.head(rows=rowNum, cols=colNum)
    assert_is_type(newFrame, H2OFrame)
    assert newFrame.dim == [rowNum, colNum], 'h2o.H2OFrame.head() command is not working.'
pyunit_utils.standalone_test(h2o_H2OFrame_head)