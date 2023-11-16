import sys
sys.path.insert(1, '../../../')
from tests import pyunit_utils
import h2o
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from io import StringIO

def h2ono_progress():
    if False:
        while True:
            i = 10
    '\n    Python API test: h2o.no_progress()\n\n    Command is verified by eyeballing the pyunit test output file and make sure the no progress bars are there.\n    Here, we will assume the command runs well if there is no error message.\n    '
    try:
        s = StringIO()
        sys.stdout = s
        h2o.no_progress()
        run_test()
        assert not s.getvalue(), 'Nothing should have been printed, instead got ' + s.getvalue()
    finally:
        sys.stdout = sys.__stdout__

def run_test():
    if False:
        i = 10
        return i + 15
    training_data = h2o.import_file(pyunit_utils.locate('smalldata/logreg/benign.csv'))
    Y = 3
    X = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10]
    model = H2OGeneralizedLinearEstimator(family='binomial', alpha=0, Lambda=1e-05)
    model.train(x=X, y=Y, training_frame=training_data)
if __name__ == '__main__':
    pyunit_utils.standalone_test(h2ono_progress)
else:
    h2ono_progress()