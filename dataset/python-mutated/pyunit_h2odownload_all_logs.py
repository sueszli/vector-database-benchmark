import sys, os
sys.path.insert(1, '../../../')
from tests import pyunit_utils
import h2o
from h2o.estimators.glm import H2OGeneralizedLinearEstimator

def h2odownload_all_logs():
    if False:
        for i in range(10):
            print('nop')
    "\n    Python API test: h2o.download_all_logs(dirname=u'.', filename=None)\n    "
    training_data = h2o.import_file(pyunit_utils.locate('smalldata/logreg/benign.csv'))
    Y = 3
    X = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10]
    model = H2OGeneralizedLinearEstimator(family='binomial', alpha=0, Lambda=1e-05)
    model.train(x=X, y=Y, training_frame=training_data)
    try:
        results_dir = pyunit_utils.locate('results')
        filename = 'logs.csv'
        dir_path = h2o.download_all_logs(results_dir, filename)
        full_path_filename = os.path.join(results_dir, filename)
        assert dir_path == full_path_filename, 'h2o.download_all_logs() command is not working.'
        assert os.path.isfile(full_path_filename), 'h2o.download_all_logs() command is not working.'
    except Exception as e:
        if 'File not found' in e.args[0]:
            print('Directory is not writable.  h2o.download_all_logs() command is not tested.')
        else:
            assert False, 'h2o.download_all_logs() command is not working.'
if __name__ == '__main__':
    pyunit_utils.standalone_test(h2odownload_all_logs)
else:
    h2odownload_all_logs()