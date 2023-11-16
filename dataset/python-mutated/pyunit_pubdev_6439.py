from h2o import H2OFrame
from tests import pyunit_utils
import sys

def pubdev_6439():
    if False:
        for i in range(10):
            print('nop')
    data = [['C1'], ['Ｘ県 Ａ市 '], ['Ｘ県 Ｂ市']]
    frame = H2OFrame(data, header=True, column_types=['enum'])
    frame_categories = frame['C1'].categories()
    print(frame_categories)
    assert len(frame_categories) == 2
    assert len(frame_categories[0]) == 6
    assert len(frame_categories[1]) == 5
    if sys.version_info[0] == 3:
        assert ''.join(data[1]) == frame_categories[0]
        assert ''.join(data[2]) == frame_categories[1]
    elif sys.version_info[0] == 2:
        assert ''.join(data[1]).decode('utf-8') == frame_categories[0]
        assert ''.join(data[2]).decode('utf-8') == frame_categories[1]
    else:
        assert False
if __name__ == '__main__':
    pyunit_utils.standalone_test(pubdev_6439)
else:
    pubdev_6439()