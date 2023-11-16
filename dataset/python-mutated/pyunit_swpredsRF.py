import sys
sys.path.insert(1, '../../../')
import h2o
from tests import pyunit_utils
from h2o.estimators.random_forest import H2ORandomForestEstimator

def swpredsRF():
    if False:
        while True:
            i = 10
    swpreds = h2o.import_file(path=pyunit_utils.locate('smalldata/gbm_test/swpreds_1000x3.csv'))
    swpreds['y'] = swpreds['y'].asfactor()
    model1 = H2ORandomForestEstimator(ntrees=50, max_depth=20, nbins=500)
    model1.train(x='X1', y='y', training_frame=swpreds)
    model1.show()
    perf1 = model1.model_performance(swpreds)
    print(perf1.auc())
    model2 = H2ORandomForestEstimator(ntrees=50, max_depth=20, nbins=500)
    model2.train(x=['X1', 'X2'], y='y', training_frame=swpreds)
    model2.show()
    perf2 = model2.model_performance(swpreds)
    print(perf2.auc())
if __name__ == '__main__':
    pyunit_utils.standalone_test(swpredsRF)
else:
    swpredsRF()