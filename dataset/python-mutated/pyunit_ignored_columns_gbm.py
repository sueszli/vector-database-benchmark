import sys
sys.path.insert(1, '../../../')
import h2o
from tests import pyunit_utils
from h2o.estimators.gbm import H2OGradientBoostingEstimator

def ignored_columns():
    if False:
        while True:
            i = 10
    airlines = h2o.upload_file(pyunit_utils.locate('smalldata/airlines/allyears2k_headers.zip'))
    airlines['Year'] = airlines['Year'].asfactor()
    airlines['Month'] = airlines['Month'].asfactor()
    airlines['DayOfWeek'] = airlines['DayOfWeek'].asfactor()
    airlines['Cancelled'] = airlines['Cancelled'].asfactor()
    airlines['FlightNum'] = airlines['FlightNum'].asfactor()
    predictors = airlines.columns[:9]
    response = 'IsDepDelayed'
    (train, valid) = airlines.split_frame(ratios=[0.8], seed=1234)
    col_list = ['DepTime', 'CRSDepTime', 'ArrTime', 'CRSArrTime']
    airlines_gbm = H2OGradientBoostingEstimator(ignored_columns=col_list, seed=1234)
    message = 'Properties x and ignored_columns cannot be specified simultaneously'
    try:
        airlines_gbm.train(x=predictors, y=response, training_frame=train, validation_frame=valid)
        assert False, 'It should throw H2OValueError: ' + message
    except Exception as e:
        assert message in e.args, 'It should throw H2OValueError: ' + message
    airlines_gbm.train(y=response, training_frame=train, validation_frame=valid)
    ignored_cols_from_model = [param_dict['actual_value'] for param_dict in airlines_gbm._model_json['parameters'] if param_dict['name'] == 'ignored_columns'][0]
    assert set(col_list) <= set(ignored_cols_from_model)
    used_columns = airlines_gbm._model_json['output']['names']
    assert set(col_list).isdisjoint(set(used_columns))
if __name__ == '__main__':
    pyunit_utils.standalone_test(ignored_columns)
else:
    ignored_columns()