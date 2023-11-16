from builtins import range
import sys
sys.path.insert(1, '../../../')
import h2o
from tests import pyunit_utils
from h2o.estimators.gbm import H2OGradientBoostingEstimator

def offset_poisson():
    if False:
        i = 10
        return i + 15
    insurance = h2o.import_file(pyunit_utils.locate('smalldata/glm_test/insurance.csv'))
    insurance['offset'] = insurance['Holders'].log()
    gbm = H2OGradientBoostingEstimator(ntrees=600, learn_rate=0.1, max_depth=1, min_rows=1, distribution='poisson', min_split_improvement=0)
    gbm.train(x=list(range(3)), y='Claims', training_frame=insurance, offset_column='offset')
    predictions = gbm.predict(insurance)
    assert abs(-2.003262 - gbm._model_json['output']['init_f']) < 1e-05, 'expected init_f to be {0}, but got {1}'.format(-2.003262, gbm._model_json['output']['init_f'])
    assert abs(49.23437 - predictions.mean().getrow()[0]) < 0.0001, 'expected prediction mean to be {0}, but got {1}'.format(49.23437, predictions.mean().getrow()[0])
    assert abs(1.077275 - predictions.min()) < 0.0001, 'expected prediction min to be {0}, but got {1}'.format(1.077275, predictions.min())
    assert abs(398.0608 - predictions.max()) < 0.01, 'expected prediction max to be {0}, but got {1}'.format(398.0608, predictions.max())
if __name__ == '__main__':
    pyunit_utils.standalone_test(offset_poisson)
else:
    offset_poisson()