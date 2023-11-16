from builtins import range
import sys
sys.path.insert(1, '../../../')
import h2o
from tests import pyunit_utils
from h2o.estimators.gbm import H2OGradientBoostingEstimator

def distribution_behavior_gbm():
    if False:
        return 10
    eco = h2o.import_file(path=pyunit_utils.locate('smalldata/gbm_test/ecology_model.csv'))
    eco_model = H2OGradientBoostingEstimator()
    eco_model.train(x=list(range(2, 13)), y='Angaus', training_frame=eco)
    cars = h2o.import_file(path=pyunit_utils.locate('smalldata/junit/cars.csv'))
    cars_model = H2OGradientBoostingEstimator()
    cars_model.train(x=list(range(3, 7)), y='cylinders', training_frame=cars)
    eco_model = H2OGradientBoostingEstimator(distribution='gaussian')
    eco_model.train(x=list(range(2, 13)), y='Angaus', training_frame=eco)
    try:
        eco_model.train(x=list(range(1, 8)), y='Method', training_frame=eco)
        assert False, 'expected an error'
    except EnvironmentError:
        assert True
    eco_model = H2OGradientBoostingEstimator(distribution='bernoulli')
    eco['Angaus'] = eco['Angaus'].asfactor()
    eco_model.train(x=list(range(2, 13)), y='Angaus', training_frame=eco)
    tree = h2o.import_file(path=pyunit_utils.locate('smalldata/junit/test_tree_minmax.csv'))
    tree_model = eco_model
    tree_model.min_rows = 1
    tree_model.train(list(range(3)), y='response', training_frame=tree)
    try:
        cars_mod = H2OGradientBoostingEstimator(distribution='bernoulli')
        cars_mod.train(x=list(range(3, 7)), y='cylinders', training_frame=cars)
        assert False, 'expected an error'
    except EnvironmentError:
        assert True
    try:
        eco_model = H2OGradientBoostingEstimator(distribution='bernoulli')
        eco_model.train(x=list(range(8)), y='Method', training_frame=eco)
        assert False, 'expected an error'
    except EnvironmentError:
        assert True
    cars['cylinders'] = cars['cylinders'].asfactor()
    cars_model = H2OGradientBoostingEstimator(distribution='multinomial')
    cars_model.train(list(range(3, 7)), y='cylinders', training_frame=cars)
    cars_model = H2OGradientBoostingEstimator(distribution='multinomial')
    cars_model.train(x=list(range(3, 7)), y='cylinders', training_frame=cars)
    eco_model = H2OGradientBoostingEstimator(distribution='multinomial')
    eco_model.train(x=list(range(8)), y='Method', training_frame=eco)
if __name__ == '__main__':
    pyunit_utils.standalone_test(distribution_behavior_gbm)
else:
    distribution_behavior_gbm()