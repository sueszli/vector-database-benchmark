import sys
import tempfile
sys.path.insert(1, '../../')
import h2o
from tests import pyunit_utils, test_plot_result_saving
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
from h2o.estimators.glm import H2OGeneralizedLinearEstimator

def varimp_plot_test():
    if False:
        print('Hello World!')
    kwargs = {}
    kwargs['server'] = True
    cars = h2o.import_file(pyunit_utils.locate('smalldata/junit/cars_20mpg.csv'))
    s = cars[0].runif()
    cars_train = cars[s <= 0.8]
    cars_valid = cars[s > 0.8]
    predictors = ['displacement', 'power', 'weight', 'acceleration', 'year']
    response_col = 'economy_20mpg'
    cars[response_col] = cars[response_col].asfactor()
    cars_rf = H2ORandomForestEstimator()
    cars_rf.train(x=predictors, y=response_col, training_frame=cars_train, validation_frame=cars_valid)
    cars_rf.varimp_plot(server=True)
    cars_rf.varimp_plot(num_of_features=2, server=True)
    tmpdir = tempfile.mkdtemp(prefix='h2o-func')
    path = '{}/plot1.png'.format(tmpdir)
    test_plot_result_saving(cars_rf.varimp_plot(server=True), '{}/plot2.png'.format(tmpdir), cars_rf.varimp_plot(server=True, save_plot_path=path), path)
    cars_gbm = H2OGradientBoostingEstimator()
    cars_gbm.train(x=predictors, y=response_col, training_frame=cars_train, validation_frame=cars_valid)
    cars_gbm.varimp_plot(server=True)
    cars_gbm.varimp_plot(num_of_features=2, server=True)
    cars_dl = H2ODeepLearningEstimator(variable_importances=True)
    cars_dl.train(x=predictors, y=response_col, training_frame=cars_train, validation_frame=cars_valid)
    cars_dl.varimp_plot(server=True)
    cars_dl.varimp_plot(num_of_features=2, server=True)
    cars_glm = H2OGeneralizedLinearEstimator()
    cars_glm.train(x=predictors, y=response_col, training_frame=cars_train, validation_frame=cars_valid)
    cars_glm.varimp_plot(server=True)
    cars_glm.varimp_plot(num_of_features=2, server=True)
if __name__ == '__main__':
    pyunit_utils.standalone_test(varimp_plot_test)
else:
    varimp_plot_test()