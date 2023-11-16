import sys
sys.path.insert(1, '../../')
import h2o
from tests import pyunit_utils

def continuous_or_categorical():
    if False:
        i = 10
        return i + 15
    df_hex = h2o.import_file(pyunit_utils.locate('smalldata/jira/hexdev_29.csv'), col_types=['enum'] * 3)
    df_hex.summary()
    assert df_hex['h1'].isfactor()
    assert df_hex['h2'].isfactor()
    assert df_hex['h3'].isfactor()
if __name__ == '__main__':
    pyunit_utils.standalone_test(continuous_or_categorical)
else:
    continuous_or_categorical()