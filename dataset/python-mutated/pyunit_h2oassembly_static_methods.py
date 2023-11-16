import sys
sys.path.insert(1, '../../')
from tests import pyunit_utils
from h2o.assembly import *
from h2o.utils.typechecks import assert_is_type

def h2oassembly_divide():
    if False:
        i = 10
        return i + 15
    '\n    Python API test: test all H2OAssembly static methods and they are:\n    H2OAssembly.divide(frame1, frame2)\n    H2OAssembly.plus(frame1, frame2)\n    H2OAssembly.multiply(frame1, frame2)\n    H2OAssembly.minus(frame1, frame2)\n    H2OAssembly.less_than(frame1, frame2)\n    H2OAssembly.less_than_equal(frame1, frame2)\n    H2OAssembly.equal_equal(frame1, frame2)\n    H2OAssembly.not_equal(frame1, frame2)\n    H2OAssembly.greater_than(frame1, frame2)\n    H2OAssembly.greater_than_equal(frame1, frame2)\n    '
    python_list1 = [[4, 4, 4, 4], [4, 4, 4, 4]]
    python_list2 = [[2, 2, 2, 2], [2, 2, 2, 2]]
    frame1 = h2o.H2OFrame(python_obj=python_list1)
    frame2 = h2o.H2OFrame(python_obj=python_list2)
    verify_results(H2OAssembly.divide(frame1, frame2), 2, 'H2OAssembly.divide()')
    verify_results(H2OAssembly.plus(frame1, frame2), 6, 'H2OAssembly.plus()')
    verify_results(H2OAssembly.multiply(frame1, frame2), 8, 'H2OAssembly.multiply()')
    verify_results(H2OAssembly.minus(frame1, frame2), 2, 'H2OAssembly.minus()')
    verify_results(H2OAssembly.less_than(frame2, frame1), 1.0, 'H2OAssembly.less_than()')
    verify_results(H2OAssembly.less_than(frame2, frame2), 0.0, 'H2OAssembly.less_than()')
    verify_results(H2OAssembly.less_than_equal(frame2, frame1), 1.0, 'H2OAssembly.less_than_equal()')
    verify_results(H2OAssembly.less_than_equal(frame2, frame2), 1.0, 'H2OAssembly.less_than_equal()')
    verify_results(H2OAssembly.equal_equal(frame2, frame1), 0.0, 'H2OAssembly.equal_equal()')
    verify_results(H2OAssembly.equal_equal(frame2, frame2), 1.0, 'H2OAssembly.equal_equal()')
    verify_results(H2OAssembly.not_equal(frame2, frame1), 1.0, 'H2OAssembly.not_equal()')
    verify_results(H2OAssembly.not_equal(frame2, frame2), 0.0, 'H2OAssembly.not_equal()')
    verify_results(H2OAssembly.greater_than(frame1, frame2), 1.0, 'H2OAssembly.greater_than()')
    verify_results(H2OAssembly.greater_than(frame2, frame2), 0.0, 'H2OAssembly.greater_than()')
    verify_results(H2OAssembly.greater_than_equal(frame1, frame2), 1.0, 'H2OAssembly.greater_than_equal()')
    verify_results(H2OAssembly.greater_than_equal(frame2, frame2), 1.0, 'H2OAssembly.greater_than_equal()')

def verify_results(resultFrame, matchValue, commandName):
    if False:
        while True:
            i = 10
    assert_is_type(resultFrame, H2OFrame)
    assert (resultFrame == matchValue).all(), commandName + ' command is not working.'
if __name__ == '__main__':
    pyunit_utils.standalone_test(h2oassembly_divide)
else:
    h2oassembly_divide()