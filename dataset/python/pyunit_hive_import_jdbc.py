#! /usr/env/python

import sys
import os
import h2o
sys.path.insert(1, os.path.join("..", "..", "..", "h2o-py"))
from h2o.utils.typechecks import (assert_is_type)
from h2o.frame import H2OFrame
from tests import pyunit_utils

def hive_import():
    connection_url = "jdbc:hive2://localhost:10000/default"
    krb_enabled = os.getenv('KRB_ENABLED', 'false').lower() == 'true'
    if krb_enabled:
        connection_url += ";auth=delegationToken"

    # import from regular table JDBC
    test_table_normal = h2o.import_hive_table(connection_url, "test_table_normal")
    assert_is_type(test_table_normal, H2OFrame)
    assert test_table_normal.nrow==3, "test_table_normal JDBC number of rows is incorrect. h2o.import_hive_table() is not working."
    assert test_table_normal.ncol==5, "test_table_normal JDBC number of columns is incorrect. h2o.import_hive_table() is not working."

    # import from partitioned table with multi format enabled JDBC
    test_table_multi_format = h2o.import_hive_table(connection_url, "test_table_multi_format", allow_multi_format=True)
    assert_is_type(test_table_multi_format, H2OFrame)
    assert test_table_multi_format.nrow==3, "test_table_multi_format JDBC number of rows is incorrect. h2o.import_hive_table() is not working."
    assert test_table_multi_format.ncol==5, "test_table_multi_format JDBC number of columns is incorrect. h2o.import_hive_table() is not working."

    # import from partitioned table with single format and partition filter JDBC
    test_table_multi_key = h2o.import_hive_table(connection_url, "test_table_multi_key", partitions=[["2017", "2"]])
    assert_is_type(test_table_multi_key, H2OFrame)
    assert test_table_multi_key.nrow==3, "test_table_multi_key JDBC number of rows is incorrect. h2o.import_hive_table() is not working."
    assert test_table_multi_key.ncol==5, "test_table_multi_key JDBC number of columns is incorrect. h2o.import_hive_table() is not working."

    # import from partitioned table with single format and special characters in partition names JDBC
    test_table_escaping = h2o.import_hive_table(connection_url, "test_table_escaping")
    assert_is_type(test_table_escaping, H2OFrame)
    assert test_table_escaping.nrow==8, "test_table_escaping JDBC number of rows is incorrect. h2o.import_hive_table() is not working."
    assert test_table_escaping.ncol==2, "test_table_escaping JDBC number of columns is incorrect. h2o.import_hive_table() is not working."


if __name__ == "__main__":
    pyunit_utils.standalone_test(hive_import)
else:
    hive_import()
