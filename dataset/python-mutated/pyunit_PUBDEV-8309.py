from builtins import range
import sys, os
from h2o.estimators import H2OGenericEstimator
sys.path.insert(1, os.path.join('..', '..', '..'))
import h2o
from tests import pyunit_utils
from h2o.estimators.gbm import H2OGradientBoostingEstimator
import tempfile

def pubdev_8309():
    if False:
        for i in range(10):
            print('nop')
    airlines = h2o.upload_file(pyunit_utils.locate('smalldata/airlines/allyears2k_headers.zip'))
    airlines['Year'] = airlines['Year'].asfactor()
    airlines['Month'] = airlines['Month'].asfactor()
    airlines['DayOfWeek'] = airlines['DayOfWeek'].asfactor()
    airlines['Cancelled'] = airlines['Cancelled'].asfactor()
    airlines['FlightNum'] = airlines['FlightNum'].asfactor()
    predictors = airlines.columns[:9]
    response = 'IsDepDelayed'
    (train, valid) = airlines.split_frame(ratios=[0.8], seed=1234)
    col_list = ['ArrTime', 'DepTime', 'CRSArrTime', 'CRSDepTime']
    airlines_gbm = H2OGradientBoostingEstimator(ignored_columns=col_list, seed=1234)
    airlines_gbm.train(y=response, training_frame=train, validation_frame=valid)
    original_model_filename = tempfile.mkdtemp()
    original_model_filename = airlines_gbm.download_mojo(original_model_filename)
    mojo_model = h2o.import_mojo(original_model_filename)
    assert isinstance(mojo_model, H2OGenericEstimator)
    assert mojo_model.params['ignored_columns'] == airlines_gbm.params['ignored_columns']
    mojo_model.params['ignored_columns']['actual'] == col_list
if __name__ == '__main__':
    pyunit_utils.standalone_test(pubdev_8309)
else:
    pubdev_8309()