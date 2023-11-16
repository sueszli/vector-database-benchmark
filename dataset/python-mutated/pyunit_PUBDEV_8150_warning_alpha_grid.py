from builtins import range
import sys
sys.path.insert(1, '../../../')
import h2o
from tests import pyunit_utils
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.grid.grid_search import H2OGridSearch
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

def grid_alpha_search():
    if False:
        print('Hello World!')
    warnNumber = 1
    hdf = h2o.upload_file(pyunit_utils.locate('smalldata/prostate/prostate_complete.csv.zip'))
    print('Testing for family: TWEEDIE')
    print('Set variables for h2o.')
    y = 'CAPSULE'
    x = ['AGE', 'RACE', 'DCAPS', 'PSA', 'VOL', 'DPROS', 'GLEASON']
    hyper_parameters = {'alpha': [0, 0.5]}
    print('Create models with lambda_search')
    buffer = StringIO()
    sys.stderr = buffer
    model_h2o_grid_search = H2OGridSearch(H2OGeneralizedLinearEstimator(family='tweedie', Lambda=0.5), hyper_parameters)
    model_h2o_grid_search.train(x=x, y=y, training_frame=hdf)
    sys.stderr = sys.__stderr__
    warn_phrase = 'Adding alpha array to hyperparameter runs slower with gridsearch.'
    try:
        assert len(buffer.buflist) == warnNumber
        print(buffer.buflist[0])
        assert warn_phrase in buffer.buflist[0]
    except:
        warns = buffer.getvalue()
        print('*** captured warning message: {0}'.format(warns))
        assert warn_phrase in warns
if __name__ == '__main__':
    pyunit_utils.standalone_test(grid_alpha_search)
else:
    grid_alpha_search()