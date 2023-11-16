import h2o
from tests import pyunit_utils

def pubdev_5167():
    if False:
        print('Hello World!')
    parquet = h2o.import_file(path=pyunit_utils.locate('smalldata/jira/pubdev-5167.parquet'))
    frame = parquet.as_data_frame()
    assert frame.shape == (1, 1)
    assert frame.loc[0][0] == 1516141767000
if __name__ == '__main__':
    pyunit_utils.standalone_test(pubdev_5167)
else:
    pubdev_5167()