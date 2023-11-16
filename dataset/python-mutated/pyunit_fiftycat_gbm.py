import sys
sys.path.insert(1, '../../../')
import h2o
from tests import pyunit_utils
from h2o.estimators.gbm import H2OGradientBoostingEstimator

def fiftycat_gbm():
    if False:
        for i in range(10):
            print('nop')
    train = h2o.import_file(path=pyunit_utils.locate('smalldata/gbm_test/50_cattest_train.csv'))
    train['y'] = train['y'].asfactor()
    model = H2OGradientBoostingEstimator(distribution='bernoulli', ntrees=10, max_depth=5, nbins=20)
    model.train(x=['x1', 'x2'], y='y', training_frame=train)
    model.show()
    test = h2o.import_file(path=pyunit_utils.locate('smalldata/gbm_test/50_cattest_test.csv'))
    predictions = model.predict(test)
    predictions.show()
    performance = model.model_performance(test)
    test_cm = performance.confusion_matrix()
    test_auc = performance.auc()
if __name__ == '__main__':
    pyunit_utils.standalone_test(fiftycat_gbm)
else:
    fiftycat_gbm()