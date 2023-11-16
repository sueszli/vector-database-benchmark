import sys, os
sys.path.insert(1, '../../../')
import h2o
from tests import pyunit_utils
from random import randint
import tempfile

def drf_leaf_node_assignment_mojo_test():
    if False:
        i = 10
        return i + 15
    problems = ['binomial', 'multinomial', 'regression']
    PROBLEM = problems[randint(0, len(problems) - 1)]
    TESTROWS = 2000
    df = pyunit_utils.random_dataset(PROBLEM, verbose=False, NTESTROWS=TESTROWS)
    train = df[TESTROWS:, :]
    test = df[:TESTROWS, :]
    x = list(set(df.names) - {'respose'})
    params = {'ntrees': 50, 'max_depth': 4}
    TMPDIR = tempfile.mkdtemp()
    my_gbm = pyunit_utils.build_save_model_generic(params, x, train, 'response', 'DRF', TMPDIR)
    MOJONAME = pyunit_utils.getMojoName(my_gbm._id)
    h2o.download_csv(test[x], os.path.join(TMPDIR, 'in.csv'))
    (pred_h2o, pred_mojo) = pyunit_utils.mojo_predict(my_gbm, TMPDIR, MOJONAME, get_leaf_node_assignment=True)
    pyunit_utils.compare_string_frames_local(pred_h2o, pred_mojo, 0.5)
if __name__ == '__main__':
    pyunit_utils.standalone_test(drf_leaf_node_assignment_mojo_test)
else:
    drf_leaf_node_assignment_mojo_test()