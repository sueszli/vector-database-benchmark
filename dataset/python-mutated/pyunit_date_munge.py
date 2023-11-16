import sys
sys.path.insert(1, '../../')
import h2o
from tests import pyunit_utils

def refine_date_col(data, col):
    if False:
        return 10
    data['Day'] = data[col].day()
    data['Month'] = data[col].month() + 1
    data['Year'] = data[col].year() + 1900
    data['WeekNum'] = data[col].week()
    data['WeekDay'] = data[col].dayOfWeek()
    data['HourOfDay'] = data[col].hour()
    data['Weekend'] = (data['WeekDay'] == 'Sun') | (data['WeekDay'] == 'Sat')
    assert data['Weekend'].min() < data['Weekend'].max()
    data['Season'] = data['Month'].cut([0, 2, 5, 7, 10, 12], ['Winter', 'Spring', 'Summer', 'Autumn', 'Winter'])

def date_munge():
    if False:
        for i in range(10):
            print('nop')
    crimes_path = pyunit_utils.locate('smalldata/chicago/chicagoCrimes10k.csv.zip')
    crimes = h2o.import_file(path=crimes_path)
    crimes.describe()
    refine_date_col(crimes, 'Date')
    crimes = crimes.drop('Date')
    crimes.describe()
if __name__ == '__main__':
    pyunit_utils.standalone_test(date_munge)
else:
    date_munge()