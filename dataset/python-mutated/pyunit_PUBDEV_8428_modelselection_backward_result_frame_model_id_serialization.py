import sys, os
sys.path.insert(1, '../../../')
import h2o
import tempfile
from tests import pyunit_utils
from h2o.estimators.model_selection import H2OModelSelectionEstimator as modelSelection

def test_modelselection_backward_serialization():
    if False:
        print('Hello World!')
    d = h2o.import_file(path=pyunit_utils.locate('smalldata/logreg/prostate.csv'))
    y = 'GLEASON'
    x = ['ID', 'AGE', 'RACE', 'CAPSULE', 'DCAPS', 'PSA', 'VOL', 'DPROS']
    model_backward = modelSelection(seed=12345, mode='backward', family='negativebinomial', link='log', alpha=0.5, lambda_=0, theta=0.01)
    model_backward.train(training_frame=d, x=x, y=y)
    model_backward2 = modelSelection(seed=12345, mode='backward', family='negativebinomial', link='log', alpha=0.5, lambda_=0, theta=0.01)
    model_backward2.train(training_frame=d, x=x, y=y)
    result = model_backward.result()
    result2 = model_backward.result()
    pyunit_utils.compare_frames_local(result[2:5], result2[2:5], prob=1.0)
    num_models = result.nrows
    one_model = h2o.get_model(result['model_id'][num_models - 1, 0])
    predict_frame = one_model.predict(d)
    tmpdir = tempfile.mkdtemp()
    file_dir = os.path.join(tmpdir, 'predict.csv')
    h2o.download_csv(predict_frame, file_dir)
    model_path_backward = model_backward.download_model(tmpdir)
    h2o.remove_all()
    d = h2o.import_file(path=pyunit_utils.locate('smalldata/logreg/prostate.csv'))
    loaded_backward_model = h2o.load_model(model_path_backward)
    result_frame_backward = loaded_backward_model.result()
    model_from_frame_backward = h2o.get_model(result_frame_backward['model_id'][num_models - 1, 0])
    pred_frame_backward = model_from_frame_backward.predict(d)
    pred_frame_model = h2o.import_file(file_dir)
    pyunit_utils.compare_frames_local(pred_frame_backward, pred_frame_model, prob=1.0)
if __name__ == '__main__':
    pyunit_utils.standalone_test(test_modelselection_backward_serialization)
else:
    test_modelselection_backward_serialization()