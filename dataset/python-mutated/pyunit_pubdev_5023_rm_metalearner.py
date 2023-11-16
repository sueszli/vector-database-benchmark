import h2o
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from tests import pyunit_utils

def pubdev_5023_rm_metalearner():
    if False:
        while True:
            i = 10
    data = h2o.import_file(pyunit_utils.locate('smalldata/higgs/higgs_train_10k.csv'))
    x = data.columns
    y = 'response'
    x.remove(y)
    data[y] = data[y].asfactor()
    gbm_h2o = H2OGradientBoostingEstimator(learn_rate=0.1, max_depth=4)
    gbm_h2o.train(x=x, y=y, training_frame=data)
    try:
        print(type(gbm_h2o.metalearner()))
        exit(1)
    except Exception as ex:
        print(ex)
if __name__ == '__main__':
    pyunit_utils.standalone_test(pubdev_5023_rm_metalearner)
else:
    pubdev_5023_rm_metalearner()