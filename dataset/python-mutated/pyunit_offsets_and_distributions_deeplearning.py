from builtins import range
import sys, os
sys.path.insert(1, os.path.join('..', '..', '..'))
import h2o
from tests import pyunit_utils
from h2o.estimators.deeplearning import H2ODeepLearningEstimator

def offsets_and_distributions():
    if False:
        print('Hello World!')
    cars = h2o.upload_file(pyunit_utils.locate('smalldata/junit/cars_20mpg.csv'))
    cars = cars[cars['economy_20mpg'].isna() == 0]
    cars['economy_20mpg'] = cars['economy_20mpg'].asfactor()
    offset = h2o.H2OFrame([[0.5]] * 398)
    offset.set_name(0, 'x1')
    cars = cars.cbind(offset)
    insurance = h2o.import_file(pyunit_utils.locate('smalldata/glm_test/insurance.csv'))
    insurance['offset'] = insurance['Holders'].log()
    dl = H2ODeepLearningEstimator(distribution='gamma')
    dl.train(x=list(range(3)), y='Claims', training_frame=insurance, offset_column='offset')
    predictions = dl.predict(insurance)
    dl = H2ODeepLearningEstimator(distribution='gaussian')
    dl.train(x=list(range(3)), y='Claims', training_frame=insurance, offset_column='offset')
    predictions = dl.predict(insurance)
    dl = H2ODeepLearningEstimator(distribution='poisson')
    dl.train(x=list(range(3)), y='Claims', training_frame=insurance, offset_column='offset')
    predictions = dl.predict(insurance)
    dl = H2ODeepLearningEstimator(distribution='tweedie')
    dl.train(x=list(range(3)), y='Claims', training_frame=insurance, offset_column='offset')
    predictions = dl.predict(insurance)
if __name__ == '__main__':
    pyunit_utils.standalone_test(offsets_and_distributions)
else:
    offsets_and_distributions()