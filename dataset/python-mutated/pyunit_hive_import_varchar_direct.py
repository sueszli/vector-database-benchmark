import sys
import os
import h2o
sys.path.insert(1, os.path.join('..', '..', '..', 'h2o-py'))
from h2o.utils.typechecks import assert_is_type
from h2o.frame import H2OFrame
from tests import pyunit_utils

def hive_import_varchar():
    if False:
        i = 10
        return i + 15
    test_table_normal = h2o.import_hive_table('default', 'AirlinesTest')
    assert_is_type(test_table_normal, H2OFrame)
    assert test_table_normal.nrow > 0
if __name__ == '__main__':
    pyunit_utils.standalone_test(hive_import_varchar)
else:
    hive_import_varchar()