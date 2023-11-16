import sys, os
sys.path.insert(1, os.path.join('..', '..', '..'))
from tests import pyunit_utils
from datetime import datetime
import h2o
import uuid
from pandas.util.testing import assert_frame_equal
import boto3
from h2o.estimators.gbm import H2OGradientBoostingEstimator
import warnings

def test_s3_model_save():
    if False:
        print('Hello World!')
    prostate = h2o.import_file(path=pyunit_utils.locate('smalldata/logreg/prostate.csv'))
    gbm = H2OGradientBoostingEstimator(pred_noise_bandwidth=0.5)
    gbm.train(y='AGE', training_frame=prostate)
    predicted_original = gbm.predict(prostate)
    for scheme in ['s3']:
        timestamp = datetime.today().utcnow().strftime('%Y%m%d-%H%M%S.%f')
        unique_suffix = str(uuid.uuid4())
        s3_dir = scheme + '://test.0xdata.com/h2o-hadoop-tests/test-save_model/' + scheme + '/exported.' + timestamp + '.' + unique_suffix
        s3_model_path = h2o.save_model(gbm, s3_dir)
        key = 'h2o-hadoop-tests/test-save_model/' + scheme + '/exported.' + timestamp + '.' + unique_suffix + '/' + gbm.model_id
        h2o.remove(gbm)
        s3 = boto3.resource('s3')
        client = boto3.client('s3')
        client.get_waiter('object_exists').wait(Bucket='test.0xdata.com', Key=key, WaiterConfig={'Delay': 2, 'MaxAttempts': 10})
        model_reloaded = h2o.load_model(s3_model_path)
        predicted_reloaded = model_reloaded.predict(prostate)
        assert_frame_equal(predicted_original.as_data_frame(), predicted_reloaded.as_data_frame())
        try:
            s3.Object(bucket_name='test.0xdata.com', key=key).delete()
        except:
            warnings.warn('Object not deleted, perform manual clean-up in h2o-hadoop-tests/test-save_model/')
if __name__ == '__main__':
    pyunit_utils.standalone_test(test_s3_model_save)
else:
    test_s3_model_save()