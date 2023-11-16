import sys, os
sys.path.insert(1, '../../')
import h2o
from tests import pyunit_utils

def sql_table():
    if False:
        while True:
            i = 10
    conn_url = os.getenv('SQLCONNURL')
    table = 'citibike20k'
    db_type = conn_url.split(':', 3)[1]
    username = password = ''
    if db_type == 'mysql':
        username = 'root'
        password = '0xdata'
    elif db_type == 'postgresql':
        username = password = 'postgres'
    citi_sql = h2o.import_sql_table(conn_url, table, username, password)
    citi_csv = h2o.import_file(pyunit_utils.locate('smalldata/demos/citibike_20k.csv'))
    py_citi_sql = citi_sql.as_data_frame(False)[1:]
    py_citi_csv = citi_csv.as_data_frame(False)[1:]
    assert first_1000_equal(py_citi_sql, py_citi_csv)
    citi_sql = h2o.import_sql_table(conn_url, table, username, password, ['starttime', 'bikeid'])
    assert citi_sql.nrow == 20000.0
    assert citi_sql.ncol == 2
    sql_select = h2o.import_sql_select(conn_url, 'SELECT starttime FROM citibike20k', username, password)
    assert sql_select.nrow == 20000.0
    assert sql_select.ncol == 1

def first_1000_equal(sql, csv):
    if False:
        for i in range(10):
            print('nop')
    if len(sql) != len(csv) or len(sql[0]) != len(csv[0]):
        return False
    for i in range(1000):
        for j in range(len(sql[i])):
            if sql[i][j] != csv[i][j] and '{0:.4f}'.format(float(sql[i][j])) != '{0:.4f}'.format(float(csv[i][j])):
                print('Different values between sql import and csv import: ', sql[i][j], csv[i][j])
                print('Sql imported row: ', sql[i])
                print('Csv imported row: ', csv[i])
                print('Row number: ', i, 'Column number; ', j)
                return False
    return True
if __name__ == '__main__':
    pyunit_utils.standalone_test(sql_table)
else:
    sql_table()