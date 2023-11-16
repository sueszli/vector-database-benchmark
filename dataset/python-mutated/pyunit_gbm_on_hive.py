import sys
import os
sys.path.insert(1, os.path.join('../../../h2o-py'))
from tests import pyunit_utils
import h2o
from numpy import isclose
from h2o.estimators.gbm import H2OGradientBoostingEstimator

def adapt_airlines(airlines_dataset):
    if False:
        while True:
            i = 10
    airlines_dataset['table_for_h2o_import.origin'] = airlines_dataset['table_for_h2o_import.origin'].asfactor()
    airlines_dataset['table_for_h2o_import.fdayofweek'] = airlines_dataset['table_for_h2o_import.fdayofweek'].asfactor()
    airlines_dataset['table_for_h2o_import.uniquecarrier'] = airlines_dataset['table_for_h2o_import.uniquecarrier'].asfactor()
    airlines_dataset['table_for_h2o_import.dest'] = airlines_dataset['table_for_h2o_import.dest'].asfactor()
    airlines_dataset['table_for_h2o_import.fyear'] = airlines_dataset['table_for_h2o_import.fyear'].asfactor()
    airlines_dataset['table_for_h2o_import.fdayofmonth'] = airlines_dataset['table_for_h2o_import.fdayofmonth'].asfactor()
    airlines_dataset['table_for_h2o_import.isdepdelayed'] = airlines_dataset['table_for_h2o_import.isdepdelayed'].asfactor()
    airlines_dataset['table_for_h2o_import.fmonth'] = airlines_dataset['table_for_h2o_import.fmonth'].asfactor()
    return airlines_dataset

def gbm_on_hive():
    if False:
        print('Hello World!')
    connection_url = 'jdbc:hive2://localhost:10000/default'
    krb_enabled = os.getenv('KRB_ENABLED', 'false').lower() == 'true'
    use_token = os.getenv('KRB_USE_TOKEN', 'false').lower() == 'true'
    if krb_enabled:
        if use_token:
            connection_url += ';auth=delegationToken'
        else:
            connection_url += ';principal=%s' % os.getenv('HIVE_PRINCIPAL', 'hive/localhost@H2O.AI')
    select_query = 'select * from airlinestest'
    username = 'hive'
    password = ''
    airlines_dataset_original = h2o.import_file(path='https://s3.amazonaws.com/h2o-public-test-data/smalldata/airlines/AirlinesTest.csv.zip')
    airlines_dataset_streaming = h2o.import_sql_select(connection_url, select_query, username, password, fetch_mode='SINGLE')
    airlines_dataset_streaming = adapt_airlines(airlines_dataset_streaming)
    pyunit_utils.compare_frames_local(airlines_dataset_original, airlines_dataset_streaming, 1)
    airlines_X_col_names = airlines_dataset_streaming.col_names[:-2]
    airlines_y_col_name = airlines_dataset_streaming.col_names[-2]
    gbm_v1 = H2OGradientBoostingEstimator(model_id='gbm_airlines_v1', seed=2000000)
    gbm_v1.train(airlines_X_col_names, airlines_y_col_name, training_frame=airlines_dataset_streaming, validation_frame=airlines_dataset_streaming)
    assert isclose(gbm_v1.auc(train=True), gbm_v1.auc(valid=True), rtol=0.0001)
if __name__ == '__main__':
    pyunit_utils.standalone_test(gbm_on_hive)
else:
    gbm_on_hive()