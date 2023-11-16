import sys, os
sys.path.insert(1, os.path.join('..', '..', '..'))
import h2o
from tests import pyunit_utils
from h2o.estimators.gbm import H2OGradientBoostingEstimator
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

def bigcat_gbm():
    if False:
        print('Hello World!')
    covtype = h2o.import_file(path=pyunit_utils.locate('smalldata/covtype/covtype.20k.data'))
    covtype[54] = covtype[54].asfactor()
    covtypeTest = h2o.import_file(path=pyunit_utils.locate('smalldata/covtype/covtype.20k.data'))
    covtypeTest[54] = covtype[54].asfactor()
    regular = H2OGradientBoostingEstimator(ntrees=10, seed=1234)
    regular.train(x=list(range(54)), y=54, training_frame=covtype)
    check_warnings(regular, 0, covtypeTest)
    covtypeTest = covtypeTest.drop(54)
    check_warnings(regular, 0, covtypeTest)
    covtypeTest = covtypeTest.drop(1)
    covtypeTest = covtypeTest.drop(1)
    check_warnings(regular, 2, covtypeTest)
    covtypeTest = h2o.import_file(path=pyunit_utils.locate('smalldata/covtype/covtype.20k.data'))
    covtypeTest[54] = covtype[54].asfactor()
    covtypeTest = covtypeTest.drop(3)
    covtypeTest = covtypeTest.drop(5)
    covtypeTest = covtypeTest.drop(7)
    check_warnings(regular, 3, covtypeTest)

def check_warnings(theModel, warnNumber, dataset):
    if False:
        print('Hello World!')
    buffer = StringIO()
    sys.stderr = buffer
    pred_h2o = theModel.predict(dataset)
    warn_phrase = 'missing'
    sys.stderr = sys.__stderr__
    try:
        assert len(buffer.buflist) == warnNumber
        if len(buffer.buflist) > 0:
            for index in range(len(buffer.buflist)):
                print('*** captured warning message: {0}'.format(buffer.buflist[index]))
                assert warn_phrase in buffer.buflist[index], 'Wrong warning message is received.'
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
            countWarns = warns.split().count(warn_phrase)
            assert countWarns == warnNumber, 'Expected number of warning: {0}.  But received {1}.'.format(warnNumber, countWarns)
if __name__ == '__main__':
    pyunit_utils.standalone_test(bigcat_gbm)
else:
    bigcat_gbm()