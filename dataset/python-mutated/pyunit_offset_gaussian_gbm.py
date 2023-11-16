from builtins import range
import sys
sys.path.insert(1, '../../../')
import h2o
from tests import pyunit_utils
from h2o.estimators.gbm import H2OGradientBoostingEstimator

def offset_gaussian():
    if False:
        while True:
            i = 10
    insurance = h2o.import_file(pyunit_utils.locate('smalldata/glm_test/insurance.csv'))
    insurance['offset'] = insurance['Holders'].log()
    gbm = H2OGradientBoostingEstimator(ntrees=600, max_depth=1, min_rows=1, learn_rate=0.1, distribution='gaussian', min_split_improvement=0)
    gbm.train(x=list(range(3)), y='Claims', training_frame=insurance, offset_column='offset')
    predictions = gbm.predict(insurance)
    assert abs(44.33016 - gbm._model_json['output']['init_f']) < 1e-05, 'expected init_f to be {0}, but got {1}'.format(44.33016, gbm._model_json['output']['init_f'])
    assert abs(1491.135 - gbm.mse()) < 0.01, 'expected mse to be {0}, but got {1}'.format(1491.135, gbm.mse())
    assert abs(49.23438 - predictions.mean().getrow()[0]) < 0.01, 'expected prediction mean to be {0}, but got {1}'.format(49.23438, predictions.mean().getrow()[0])
    assert abs(-45.5720659304 - predictions.min()) < 0.01, 'expected prediction min to be {0}, but got {1}'.format(-45.5720659304, predictions.min())
    assert abs(207.387 - predictions.max()) < 0.01, 'expected prediction max to be {0}, but got {1}'.format(207.387, predictions.max())
if __name__ == '__main__':
    pyunit_utils.standalone_test(offset_gaussian)
else:
    offset_gaussian()