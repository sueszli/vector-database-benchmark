import h2o
from tests import pyunit_utils
'\nChecking corner cases when initializing H2OFrame\n\n'

def test_new_empty_frame():
    if False:
        for i in range(10):
            print('nop')
    fr = h2o.H2OFrame()
    assert not fr._has_content()
    fr.describe()

def test_new_frame_with_empty_list():
    if False:
        for i in range(10):
            print('nop')
    fr = h2o.H2OFrame([])
    assert_empty(fr)
    fr.describe()

def test_new_frame_with_empty_tuple():
    if False:
        print('Hello World!')
    fr = h2o.H2OFrame(())
    assert_empty(fr)
    fr.describe()

def test_new_frame_with_empty_nested_list():
    if False:
        while True:
            i = 10
    fr = h2o.H2OFrame([[]])
    assert_empty(fr)
    fr.describe()

def test_new_frame_with_empty_dict():
    if False:
        return 10
    fr = h2o.H2OFrame({})
    assert_empty(fr)
    fr.describe()

def assert_empty(frame):
    if False:
        print('Hello World!')
    assert frame._has_content()
    assert frame.nrows == 0
    assert frame.ncols == 1
if __name__ == '__main__':
    pyunit_utils.standalone_test(test_new_empty_frame)
    pyunit_utils.standalone_test(test_new_frame_with_empty_list)
    pyunit_utils.standalone_test(test_new_frame_with_empty_tuple)
    pyunit_utils.standalone_test(test_new_frame_with_empty_nested_list)
    pyunit_utils.standalone_test(test_new_frame_with_empty_dict)
else:
    test_new_empty_frame()
    test_new_frame_with_empty_list()
    test_new_frame_with_empty_tuple()
    test_new_frame_with_empty_nested_list()
    test_new_frame_with_empty_dict()