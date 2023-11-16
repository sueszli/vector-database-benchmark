import sys
sys.path.insert(1, '../../')
import h2o
from tests import pyunit_utils
from h2o.estimators.glm import H2OGeneralizedLinearEstimator

def std_coef_plot_test():
    if False:
        return 10
    kwargs = {}
    kwargs['server'] = True
    cars = h2o.import_file(pyunit_utils.locate('smalldata/junit/cars_20mpg.csv'))
    s = cars[0].runif()
    cars_train = cars[s <= 0.8]
    cars_valid = cars[s > 0.8]
    predictors = ['displacement', 'power', 'weight', 'acceleration', 'year']
    response_col = 'economy_20mpg'
    cars[response_col] = cars[response_col].asfactor()
    cars_glm = H2OGeneralizedLinearEstimator()
    cars_glm.train(x=predictors, y=response_col, training_frame=cars_train, validation_frame=cars_valid)
    cars_glm.std_coef_plot(server=True)
    cars_glm.std_coef_plot(num_of_features=2, server=True)
if __name__ == '__main__':
    pyunit_utils.standalone_test(std_coef_plot_test)
else:
    std_coef_plot_test()