import sys
sys.path.insert(1, '../../../')
import h2o
from tests import pyunit_utils
from h2o.utils.typechecks import assert_is_type
from random import randrange
import numpy as np

def h2o_H2OFrame_tolower():
    if False:
        while True:
            i = 10
    '\n    Python API test: h2o.frame.H2OFrame.tolower()\n    '
    frame = h2o.import_file(path=pyunit_utils.locate('smalldata/iris/iris.csv'))
    frame['C5'] = frame['C5'].tolower()
    assert (frame['C5'] == 'iris-setosa').sum() == 50, 'h2o.H2OFrame.tolower() command is not working.'
pyunit_utils.standalone_test(h2o_H2OFrame_tolower)