import sys
sys.path.insert(1, '../../')
import h2o
from tests import pyunit_utils

def pyunit_colname_uniop():
    if False:
        print('Hello World!')
    dataframe = {'A': [1, 0, 3, 4], 'B': [5, 6, -6, -1], 'C': [-4, -6, -7, 8]}
    frame = h2o.H2OFrame(dataframe)
    frame_asin = frame.asin()
    assert set(frame.names) == {'A', 'B', 'C'}, 'Expected original colnames to remain the same after uniop operation'
    assert ['asin(%s)' % name for name in frame.names] == frame_asin.names, 'Expected equal col names after uniop operation'
    frame_asin.refresh()
    assert ['asin(%s)' % name for name in frame.names] == frame_asin.names, 'Expected equal col names after uniop operation'
    assert frame_asin.types == {'asin(A)': 'real', 'asin(B)': 'real', 'asin(C)': 'int'}, 'Expect equal col types afteruniop operation'
if __name__ == '__main__':
    pyunit_utils.standalone_test(pyunit_colname_uniop)
else:
    pyunit_colname_uniop()