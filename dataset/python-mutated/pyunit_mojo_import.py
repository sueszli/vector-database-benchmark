import h2o
import tempfile
from h2o.estimators import H2OGradientBoostingEstimator, H2OGenericEstimator
from tests import pyunit_utils
import os
from pandas.testing import assert_frame_equal

def mojo_convenience():
    if False:
        return 10
    airlines = h2o.import_file(path=pyunit_utils.locate('smalldata/testng/airlines_train.csv'))
    model = H2OGradientBoostingEstimator(ntrees=1)
    model.train(x=['Origin', 'Dest'], y='IsDepDelayed', training_frame=airlines)
    original_model_filename = tempfile.mkdtemp()
    original_model_filename = model.save_mojo(original_model_filename)
    mojo_model = h2o.import_mojo(original_model_filename)
    assert isinstance(mojo_model, H2OGenericEstimator)
    predictions = mojo_model.predict(airlines)
    assert predictions is not None
    assert predictions.nrows == 24421
    try:
        pyunit_utils.set_forbidden_paths([original_model_filename])
        original_model_filename = model.download_mojo(original_model_filename)
        mojo_model = h2o.upload_mojo(original_model_filename)
        assert isinstance(mojo_model, H2OGenericEstimator)
        predictions = mojo_model.predict(airlines)
        assert predictions is not None
        assert predictions.nrows == 24421
    finally:
        pyunit_utils.clear_forbidden_paths()
    pojo_directory = os.path.join(pyunit_utils.locate('results'), model.model_id + '.java')
    pojo_path = model.download_pojo(path=pojo_directory)
    mojo2_model = h2o.import_mojo(pojo_path)
    predictions2 = mojo2_model.predict(airlines)
    assert predictions2 is not None
    assert predictions2.nrows == 24421
    assert_frame_equal(predictions.as_data_frame(), predictions2.as_data_frame())
if __name__ == '__main__':
    pyunit_utils.standalone_test(mojo_convenience)
else:
    mojo_convenience()