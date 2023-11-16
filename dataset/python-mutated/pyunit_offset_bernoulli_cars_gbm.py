from builtins import range
import sys
sys.path.insert(1, '../../../')
import h2o
from tests import pyunit_utils
from h2o.estimators.gbm import H2OGradientBoostingEstimator

def offset_bernoulli_cars():
    if False:
        print('Hello World!')
    cars = h2o.upload_file(pyunit_utils.locate('smalldata/junit/cars_20mpg.csv'))
    cars = cars[cars['economy_20mpg'].isna() == 0]
    cars['economy_20mpg'] = cars['economy_20mpg'].asfactor()
    offset = h2o.H2OFrame([[0.5]] * 398)
    offset.set_names(['x1'])
    cars = cars.cbind(offset)
    gbm = H2OGradientBoostingEstimator(ntrees=1, max_depth=1, min_rows=1, learn_rate=1)
    gbm.train(x=list(range(2, 8)), y='economy_20mpg', training_frame=cars, offset_column='x1')
    predictions = gbm.predict(cars)
    assert abs(-0.1041234 - gbm._model_json['output']['init_f']) < 1e-06, 'expected init_f to be {0}, but got {1}'.format(-0.1041234, gbm._model_json['output']['init_f'])
    assert abs(0.577326 - predictions[:, 2].mean().getrow()[0]) < 1e-06, 'expected prediction mean to be {0}, but got {1}'.format(0.577326, predictions[:, 2].mean().getrow()[0])
    assert abs(0.1621461 - predictions[:, 2].min()) < 1e-06, 'expected prediction min to be {0}, but got {1}'.format(0.1621461, predictions[:, 2].min())
    assert abs(0.8506528 - predictions[:, 2].max()) < 1e-06, 'expected prediction max to be {0}, but got {1}'.format(0.8506528, predictions[:, 2].max())
if __name__ == '__main__':
    pyunit_utils.standalone_test(offset_bernoulli_cars)
else:
    offset_bernoulli_cars()