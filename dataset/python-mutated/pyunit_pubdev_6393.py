from h2o import H2OFrame
from tests import pyunit_utils

def pubdev_6393():
    if False:
        return 10
    locations = [['location'], ['�Ｘ県 Ａ市 '], ['Ｘ県 Ｂ市']]
    frame = H2OFrame(locations, header=True, column_types=['enum'])
    assert frame.ncols == 1
    assert frame.nrows == len(locations) - 1
    frame_categories = frame['location'].categories()
    print(frame_categories)
    frame_converted = frame['location'].ascharacter().asfactor()
    assert frame_converted.ncols == 1
    assert frame_converted.nrows == len(locations) - 1
    frame_converted_categories = frame_converted.categories()
    print(frame_converted_categories)
    for i in range(0, len(frame_converted_categories)):
        assert frame_categories[i] == frame_converted_categories[i]
if __name__ == '__main__':
    pyunit_utils.standalone_test(pubdev_6393)
else:
    pubdev_6393()