from builtins import zip
from builtins import range
import sys
sys.path.insert(1, '../../')
import h2o
from tests import pyunit_utils

def javapredict_pubdev_4531():
    if False:
        print('Hello World!')
    train = h2o.upload_file(pyunit_utils.locate('smalldata/logreg/prostate_train_null_column_name.csv'))
    test = h2o.upload_file(pyunit_utils.locate('smalldata/logreg/prostate_train_null_column_name.csv'))
    params = {'ntrees': 2, 'seed': 42, 'training_frame': train, 'distribution': 'bernoulli'}
    train['CAPSULE'] = train['CAPSULE'].asfactor()
    test['CAPSULE'] = test['CAPSULE'].asfactor()
    print('Parameter list:')
    for (k, v) in zip(list(params.keys()), list(params.values())):
        print('{0}, {1}'.format(k, v))
    x = list(range(0, train.ncol))
    y = 'CAPSULE'
    pyunit_utils.javapredict('gbm', 'class', train, test, x, y, **params)
    pyunit_utils.javapredict('gbm', 'class', train, test, x, y, separator='|', setInvNumNA=True, **params)
    pyunit_utils.javapredict('gbm', 'class', train, test, x, y, separator='\\|', setInvNumNA=True, **params)
    pyunit_utils.javapredict('gbm', 'class', train, test, x, y, separator='\\|', setInvNumNA=True, **params)
    pyunit_utils.javapredict('gbm', 'class', train, test, x, y, separator='@', setInvNumNA=True, **params)
if __name__ == '__main__':
    pyunit_utils.standalone_test(javapredict_pubdev_4531)
else:
    javapredict_pubdev_4531()