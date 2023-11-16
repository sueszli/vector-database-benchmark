import sys
sys.path.insert(1, '../../../')
import h2o
from tests import pyunit_utils
from h2o.grid.grid_search import H2OGridSearch
from h2o.estimators.random_forest import H2ORandomForestEstimator

def fiftycatRF():
    if False:
        return 10
    train = h2o.import_file(path=pyunit_utils.locate('smalldata/gbm_test/50_cattest_train.csv'))
    train['y'] = train['y'].asfactor()
    hyper_parameters = {'ntrees': [10, 50], 'max_depth': [20, 10]}
    model = H2OGridSearch(H2ORandomForestEstimator, hyper_params=hyper_parameters)
    print(model)
    model.train(x=['x1', 'x2'], y='y', training_frame=train)
    model.show()
    model.summary()
    for m in model:
        assert isinstance(m, H2ORandomForestEstimator)
if __name__ == '__main__':
    pyunit_utils.standalone_test(fiftycatRF)
else:
    fiftycatRF()