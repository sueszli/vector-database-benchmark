import sys, os
sys.path.insert(1, '../../../')
import h2o
from tests import pyunit_utils
from random import randint
from random import uniform
import tempfile

def glm_ordinal_mojo_pojo():
    if False:
        return 10
    params = set_params()
    NTESTROWS = 200
    PROBLEM = 'multinomial'
    df = pyunit_utils.random_dataset(PROBLEM, NTESTROWS, seed=12345)
    train = df[NTESTROWS:, :]
    test = df[:NTESTROWS, :]
    x = list(set(df.names) - {'response'})
    TMPDIR = tempfile.mkdtemp()
    glmOrdinalModel = pyunit_utils.build_save_model_generic(params, x, train, 'response', 'glm', TMPDIR)
    MOJONAME = pyunit_utils.getMojoName(glmOrdinalModel._id)
    h2o.download_csv(test[x], os.path.join(TMPDIR, 'in.csv'))
    (pred_h2o, pred_mojo) = pyunit_utils.mojo_predict(glmOrdinalModel, TMPDIR, MOJONAME)
    h2o.download_csv(pred_h2o, os.path.join(TMPDIR, 'h2oPred.csv'))
    pred_pojo = pyunit_utils.pojo_predict(glmOrdinalModel, TMPDIR, MOJONAME)
    print('Comparing mojo predict and h2o predict...')
    pyunit_utils.compare_frames_local(pred_h2o, pred_mojo, 0.1, tol=1e-10)
    print('Comparing pojo predict and h2o predict...')
    pyunit_utils.compare_frames_local(pred_mojo, pred_pojo, 0.1, tol=1e-10)

def set_params():
    if False:
        for i in range(10):
            print('nop')
    missingValues = ['MeanImputation']
    missing_values = missingValues[randint(0, len(missingValues) - 1)]
    reg = 1.0 / 250000.0
    params = {'missing_values_handling': missing_values, 'family': 'ordinal', 'alpha': [0.5], 'lambda_': [reg], 'obj_reg': reg}
    if uniform(0.0, 1.0) > 0.5:
        params['solver'] = 'GRADIENT_DESCENT_SQERR'
    print(params)
    return params
if __name__ == '__main__':
    pyunit_utils.standalone_test(glm_ordinal_mojo_pojo)
else:
    glm_ordinal_mojo_pojo()