import sys
sys.path.insert(1, '../../../')
import h2o
from tests import pyunit_utils
from h2o.estimators.random_forest import H2ORandomForestEstimator

def fiftycatRF():
    if False:
        while True:
            i = 10
    train = h2o.import_file(path=pyunit_utils.locate('smalldata/gbm_test/50_cattest_train.csv'))
    train['y'] = train['y'].asfactor()
    model = H2ORandomForestEstimator(ntrees=50, max_depth=20, nbins=500)
    model.train(x=['x1', 'x2'], y='y', training_frame=train)
    test = h2o.import_file(path=pyunit_utils.locate('smalldata/gbm_test/50_cattest_test.csv'))
    preds = model.predict(test)
    preds.head()
    perf = model.model_performance(test)
    perf.show()
    cm = perf.confusion_matrix()
    print(cm)
if __name__ == '__main__':
    pyunit_utils.standalone_test(fiftycatRF)
else:
    fiftycatRF()