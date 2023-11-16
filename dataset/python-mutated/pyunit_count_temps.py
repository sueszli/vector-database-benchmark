import sys, os
sys.path.insert(1, os.path.join('..', '..'))
import h2o
from tests import pyunit_utils

def date_munge():
    if False:
        while True:
            i = 10
    crimes_path = pyunit_utils.locate('smalldata/chicago/chicagoCrimes10k.csv.zip')
    hc = h2o.connection()
    assert hc.session_id
    tmps0 = pyunit_utils.temp_ctr()
    crimes = h2o.import_file(path=crimes_path, destination_frame='xxxx_crimes')
    rest1 = hc.requests_count
    crimes['Day'] = crimes['Date'].day()
    crimes['Month'] = crimes['Date'].month() + 1
    crimes['Year'] = crimes['Date'].year() + 1900
    crimes['WeekNum'] = crimes['Date'].week()
    crimes['WeekDay'] = crimes['Date'].dayOfWeek()
    crimes['HourOfDay'] = crimes['Date'].hour()
    print('# of REST calls used: %d' % (hc.requests_count - rest1))
    crimes['Weekend'] = (crimes['WeekDay'] == 'Sun') | (crimes['WeekDay'] == 'Sat')
    print('# of REST calls used: %d' % (hc.requests_count - rest1))
    crimes['Season'] = crimes['Month'].cut([0, 2, 5, 7, 10, 12], ['Winter', 'Spring', 'Summer', 'Autumn', 'Winter'])
    print('# of REST calls used: %d' % (hc.requests_count - rest1))
    crimes = crimes.drop('Date')
    print('# of REST calls used: %d' % (hc.requests_count - rest1))
    crimes.describe()
    print('# of REST calls used: %d' % (hc.requests_count - rest1))
    ntmps = pyunit_utils.temp_ctr() - tmps0
    nrest = pyunit_utils.rest_ctr() - rest1
    print('Number of temps used: %d' % ntmps)
    print('Number of RESTs used: %d' % nrest)
    assert ntmps == 8
    assert nrest == 2

def test_date_munge():
    if False:
        for i in range(10):
            print('nop')
    saved_flag = h2o.is_expr_optimizations_enabled()
    try:
        h2o.enable_expr_optimizations(False)
        date_munge()
    finally:
        h2o.enable_expr_optimizations(saved_flag)
if __name__ == '__main__':
    pyunit_utils.standalone_test(test_date_munge)
else:
    test_date_munge()