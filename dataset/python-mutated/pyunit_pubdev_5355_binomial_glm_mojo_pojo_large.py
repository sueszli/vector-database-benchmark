import sys, os
sys.path.insert(1, '../../../')
import h2o
from tests import pyunit_utils
from random import randint
import tempfile

def glm_binomial_mojo_pojo():
    if False:
        i = 10
        return i + 15
    h2o.remove_all()
    NTESTROWS = 200
    PROBLEM = 'binomial'
    params = set_params()
    df = pyunit_utils.random_dataset(PROBLEM)
    train = df[NTESTROWS:, :]
    test = df[:NTESTROWS, :]
    x = list(set(df.names) - {'response'})
    TMPDIR = tempfile.mkdtemp()
    glmBinomialModel = pyunit_utils.build_save_model_generic(params, x, train, 'response', 'glm', TMPDIR)
    MOJONAME = pyunit_utils.getMojoName(glmBinomialModel._id)
    h2o.download_csv(test[x], os.path.join(TMPDIR, 'in.csv'))
    (pred_h2o, pred_mojo) = pyunit_utils.mojo_predict(glmBinomialModel, TMPDIR, MOJONAME)
    h2o.download_csv(pred_h2o, os.path.join(TMPDIR, 'h2oPred.csv'))
    pred_pojo = pyunit_utils.pojo_predict(glmBinomialModel, TMPDIR, MOJONAME)
    print('Comparing mojo predict and h2o predict...')
    pyunit_utils.compare_frames_local(pred_h2o, pred_mojo, 0.1, tol=1e-10)
    print('Comparing pojo predict and h2o predict...')
    pyunit_utils.compare_frames_local(pred_mojo, pred_pojo, 0.1, tol=1e-10)

def set_params():
    if False:
        return 10
    missingValues = ['MeanImputation']
    missing_values = missingValues[randint(0, len(missingValues) - 1)]
    params = {'missing_values_handling': missing_values, 'family': 'binomial'}
    print(params)
    return params
if __name__ == '__main__':
    pyunit_utils.standalone_test(glm_binomial_mojo_pojo)
else:
    glm_binomial_mojo_pojo()