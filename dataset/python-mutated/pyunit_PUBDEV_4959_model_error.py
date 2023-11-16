from builtins import range
import sys, os
sys.path.insert(1, '../../../')
import h2o
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from tests import pyunit_utils
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

def test_benign():
    if False:
        for i in range(10):
            print('nop')
    training_data = h2o.import_file(pyunit_utils.locate('smalldata/gbm_test/BostonHousing.csv'))
    Y = 13
    X = list(range(13))
    check_warnings('Wendy Wong and Nidhi Mehta', X, Y, 0, training_data)
    check_warnings('Wendy / Wong / and / Nidhi / Mehta', X, Y, 1, training_data)
    check_warnings(' test -  june/fifteenth ', X, Y, 1, training_data)

def check_warnings(model_id, X, Y, warnNumber, training_data):
    if False:
        return 10
    buffer = StringIO()
    sys.stderr = buffer
    model = H2OGeneralizedLinearEstimator(family='Gaussian', nfolds=2, fold_assignment='Modulo', keep_cross_validation_predictions=True, model_id=model_id)
    model.train(x=X, y=Y, training_frame=training_data)
    warn_phrase = 'UserWarning'
    warn_string_of_interest = 'slash (/) found'
    sys.stderr = sys.__stderr__
    try:
        assert len(buffer.buflist) == warnNumber
        if len(buffer.buflist) > 0:
            for index in range(len(buffer.buflist)):
                print('*** captured warning message: {0}'.format(buffer.buflist[index]))
                assert warn_phrase in buffer.buflist[index] and warn_string_of_interest in buffer.buflist[index]
    except:
        if warnNumber == 0:
            try:
                warns = buffer.getvalue()
                assert False, 'Warning not expected but received...'
            except:
                assert True, 'Warning not expected but received...'
        else:
            warns = buffer.getvalue()
            print('*** captured warning message: {0}'.format(warns))
            countWarns = warns.split().count('slash')
            assert countWarns == warnNumber, 'Expected number of warnings: {0}, but received {1}.'.format(warnNumber, countWarns)
if __name__ == '__main__':
    pyunit_utils.standalone_test(test_benign)
else:
    test_benign()