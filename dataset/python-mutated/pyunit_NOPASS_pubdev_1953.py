import sys
sys.path.insert(1, '../../')
import h2o
from tests import pyunit_utils

def pubdev_1953():
    if False:
        while True:
            i = 10
    predictors = ['DayOfWeek', 'WC1', 'start station name', 'Temperature (C)', 'Days', 'Month', 'Humidity Fraction', 'Rain (mm)', 'Dew Point (C)']
    train = h2o.import_file(pyunit_utils.locate('smalldata/glm_test/citibike_small_train.csv'))
    test = h2o.import_file(pyunit_utils.locate('smalldata/glm_test/citibike_small_test.csv'))
    glm0 = h2o.glm(x=train[predictors], y=train['bikes'], validation_x=test[predictors], validation_y=test['bikes'], family='poisson')
if __name__ == '__main__':
    pyunit_utils.standalone_test(pubdev_1953)
else:
    pubdev_1953()