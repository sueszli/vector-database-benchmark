from builtins import range
import sys, os
sys.path.insert(1, os.path.join('..', '..', '..'))
import h2o
from tests import pyunit_utils
from h2o.estimators.deeplearning import H2OAutoEncoderEstimator

def anomaly():
    if False:
        for i in range(10):
            print('nop')
    print('Deep Learning Anomaly Detection MNIST')
    train = h2o.import_file(pyunit_utils.locate('bigdata/laptop/mnist/train.csv.gz'))
    test = h2o.import_file(pyunit_utils.locate('bigdata/laptop/mnist/test.csv.gz'))
    predictors = list(range(0, 784))
    resp = 784
    train = train[predictors]
    test = test[predictors]
    ae_model = H2OAutoEncoderEstimator(activation='Tanh', hidden=[2], l1=1e-05, ignore_const_cols=False, epochs=1)
    ae_model.train(x=predictors, training_frame=train)
    test_rec_error = ae_model.anomaly(test)
    test_recon = ae_model.predict(test)
if __name__ == '__main__':
    pyunit_utils.standalone_test(anomaly)
else:
    anomaly()