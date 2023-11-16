import sys
sys.path.insert(1, '../../../')
import h2o
from tests import pyunit_utils
from h2o.estimators.gbm import H2OGradientBoostingEstimator

def swpreds_gbm():
    if False:
        while True:
            i = 10
    swpreds = h2o.import_file(path=pyunit_utils.locate('smalldata/gbm_test/swpreds_1000x3.csv'))
    swpreds['y'] = swpreds['y'].asfactor()
    h2o_gbm_model1 = H2OGradientBoostingEstimator(distribution='bernoulli', ntrees=50, max_depth=20, nbins=500)
    h2o_gbm_model1.train(x='X1', y='y', training_frame=swpreds)
    h2o_gbm_model1.show()
    h2o_gbm_perf1 = h2o_gbm_model1.model_performance(swpreds)
    h2o_auc1 = h2o_gbm_perf1.auc()
    h2o_gbm_model2 = h2o_gbm_model1
    h2o_gbm_model2.train(x=['X1', 'X2'], y='y', training_frame=swpreds)
    h2o_gbm_model2.show()
    h2o_gbm_perf2 = h2o_gbm_model2.model_performance(swpreds)
    h2o_auc2 = h2o_gbm_perf2.auc()
if __name__ == '__main__':
    pyunit_utils.standalone_test(swpreds_gbm)
else:
    swpreds_gbm()