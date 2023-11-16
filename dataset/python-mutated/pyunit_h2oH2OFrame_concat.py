import sys
sys.path.insert(1, '../../../')
import h2o
from tests import pyunit_utils
from h2o.utils.typechecks import assert_is_type
from h2o.frame import H2OFrame

def h2o_H2OFrame_concat():
    if False:
        for i in range(10):
            print('nop')
    '\n    Python API test: h2o.frame.H2OFrame.concat(frames, axis=1)\n\n    Copied from pyunit_concat.py\n    '
    df1 = h2o.create_frame(integer_fraction=1, binary_fraction=0, categorical_fraction=0, seed=1)
    df2 = h2o.create_frame(integer_fraction=1, binary_fraction=0, categorical_fraction=0, seed=2)
    df3 = h2o.create_frame(integer_fraction=1, binary_fraction=0, categorical_fraction=0, seed=3)
    df123 = df1.concat([df2, df3])
    assert_is_type(df123, H2OFrame)
    assert df123.shape == (df1.nrows, df1.ncols + df2.ncols + df3.ncols), 'h2o.H2OFrame.concat command is not working.'
    df123_row = df1.concat([df2, df3], axis=0)
    assert_is_type(df123_row, H2OFrame)
    assert df123_row.shape == (df1.nrows + df2.nrows + df3.nrows, df1.ncols), 'h2o.H2OFrame.concat command is not working.'
pyunit_utils.standalone_test(h2o_H2OFrame_concat)