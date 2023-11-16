import sys
sys.path.insert(1, '../../../')
from tests import pyunit_utils
import h2o
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.utils.typechecks import assert_is_type
import inspect

def h2oshow_progress():
    if False:
        while True:
            i = 10
    '\n    Python API test: h2o.show_progress()\n\n    Command is verified by eyeballing the pyunit test output file and make sure the progress bars are there.\n    Here, we will assume the command runs well if there is no error message.\n    '
    try:
        s = StringIO()
        sys.stdout = s
        h2o.show_progress()
        training_data = h2o.upload_file(pyunit_utils.locate('smalldata/logreg/benign.csv'))
        Y = 3
        X = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10]
        model = H2OGeneralizedLinearEstimator(family='binomial', alpha=0, Lambda=1e-05)
        model.train(x=X, y=Y, training_frame=training_data)
        sys.stdout = sys.__stdout__
        assert 'progress' in s.getvalue() and '100%' in s.getvalue(), 'h2o.show_progress() command is not working.'
    except Exception as e:
        sys.stdout = sys.__stdout__
        assert_is_type(e, AttributeError)
        assert 'encoding' in e.args[0], 'h2o.show_progress() command is not working.'
        allargs = inspect.getfullargspec(h2o.show_progress)
        assert len(allargs.args) == 0, 'h2o.show_progress() should have no arguments!'
if __name__ == '__main__':
    pyunit_utils.standalone_test(h2oshow_progress)
else:
    h2oshow_progress()