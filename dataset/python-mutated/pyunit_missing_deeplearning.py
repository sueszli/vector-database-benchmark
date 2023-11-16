from builtins import range
import sys, os
sys.path.insert(1, os.path.join('..', '..', '..'))
import h2o
from tests import pyunit_utils
from h2o.estimators.deeplearning import H2ODeepLearningEstimator

def missing():
    if False:
        while True:
            i = 10
    missing_ratios = [0, 0.1, 0.25, 0.5, 0.75, 0.99]
    errors = [0, 0, 0, 0, 0, 0]
    for i in range(len(missing_ratios)):
        data = h2o.upload_file(pyunit_utils.locate('smalldata/junit/weather.csv'))
        data[15] = data[15].asfactor()
        data[16] = data[16].asfactor()
        data[17] = data[17].asfactor()
        data[18] = data[18].asfactor()
        data[19] = data[19].asfactor()
        data[21] = data[21].asfactor()
        data[23] = data[23].asfactor()
        print('For missing {0}%'.format(missing_ratios[i] * 100))
        if missing_ratios[i] > 0:
            resp = data[23]
            pred = data[:, list(range(23)) + list(range(24, data.ncol))]
            data_missing = pred.insert_missing_values(fraction=missing_ratios[i])
            data_fin = data_missing.cbind(resp)
        else:
            data_fin = data
        ratio = data_fin[0].runif()
        train = data_fin[ratio <= 0.75]
        test = data_fin[ratio > 0.75]
        hh = H2ODeepLearningEstimator(epochs=5, reproducible=True, seed=12345, activation='RectifierWithDropout', l1=1e-05, input_dropout_ratio=0.2)
        hh.train(x=list(range(2, 22)), y=23, training_frame=train, validation_frame=test)
        errors[i] = hh.error()[0][1]
    for i in range(len(missing_ratios)):
        print('missing ratio: {0}% --> classification error: {1}'.format(missing_ratios[i] * 100, errors[i]))
    assert sum(errors) < 2.2, 'Sum of classification errors is too large!'
if __name__ == '__main__':
    pyunit_utils.standalone_test(missing)
else:
    missing()