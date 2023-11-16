import sys, os
sys.path.insert(1, '../../../')
import h2o
from tests import pyunit_utils
import tempfile

def glm_gamma_offset_mojo():
    if False:
        return 10
    train = h2o.import_file(path=pyunit_utils.locate('smalldata/prostate/prostate_complete.csv.zip'))
    y = 'GLEASON'
    x = ['AGE', 'RACE', 'CAPSULE', 'DCAPS', 'PSA', 'VOL', 'DPROS']
    offset = pyunit_utils.random_dataset_real_only(train.nrow, 1, realR=3, misFrac=0, randSeed=12345)
    train = train.cbind(offset)
    offset_name = train.names[train.ncol - 1]
    x_offset = ['AGE', 'RACE', 'CAPSULE', 'DCAPS', 'PSA', 'VOL', 'DPROS', offset_name]
    params = {'offset_column': offset_name, 'family': 'negativebinomial', 'link': 'log', 'alpha': 0.5, 'lambda_': 0, 'theta': 0.01}
    tmpdir = tempfile.mkdtemp()
    glm_gamma_model = pyunit_utils.build_save_model_generic(params, x, train, y, 'glm', tmpdir)
    MOJONAME = pyunit_utils.getMojoName(glm_gamma_model._id)
    h2o.download_csv(train[x_offset], os.path.join(tmpdir, 'in.csv'))
    (pred_h2o, pred_mojo) = pyunit_utils.mojo_predict(glm_gamma_model, tmpdir, MOJONAME)
    h2o.download_csv(pred_h2o, os.path.join(tmpdir, 'h2oPred.csv'))
    print('Comparing mojo predict and h2o predict...')
    pyunit_utils.compare_frames_local(pred_h2o, pred_mojo, 0.1, tol=1e-10)
if __name__ == '__main__':
    pyunit_utils.standalone_test(glm_gamma_offset_mojo)
else:
    glm_gamma_offset_mojo()