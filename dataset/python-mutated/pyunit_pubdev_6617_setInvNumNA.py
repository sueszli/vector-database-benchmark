import sys
import tempfile
import shutil
import time
import os
import pandas
sys.path.insert(1, '../../')
import h2o
from tests import pyunit_utils
from h2o.estimators.glm import H2OGeneralizedLinearEstimator

def test_setInvNumNA():
    if False:
        print('Hello World!')
    train = h2o.import_file(pyunit_utils.locate('smalldata/glm_test/pubdev_6617_setInvNumNA_train.csv'))
    testdata = pyunit_utils.locate('smalldata/glm_test/pubdev_6617_setInvNumNA_test.csv')
    testdataModel = h2o.import_file(pyunit_utils.locate('smalldata/glm_test/pubdev_6617_setInvNumNA_test_model.csv'))
    response = 'C2'
    x = ['C1']
    params = {'missing_values_handling': 'MeanImputation', 'family': 'gaussian'}
    tmpdir = tempfile.mkdtemp()
    glmMultinomialModel = pyunit_utils.build_save_model_generic(params, x, train, response, 'glm', tmpdir)
    mojoname = pyunit_utils.getMojoName(glmMultinomialModel._id)
    mojoLoco = os.path.join(tmpdir, mojoname) + '.zip'
    mojoOut = os.path.join(tmpdir, 'mojo_out.csv')
    genJarDir = str.split(os.path.realpath('__file__'), '/')
    genJarDir = '/'.join(genJarDir[0:genJarDir.index('h2o-py')])
    jarpath = os.path.join(genJarDir, 'h2o-assemblies/genmodel/build/libs/genmodel.jar')
    mojoPredict = h2o.mojo_predict_csv(input_csv_path=testdata, mojo_zip_path=mojoLoco, output_csv_path=mojoOut, genmodel_jar_path=jarpath, verbose=True, setInvNumNA=True)
    modelPred = glmMultinomialModel.predict(testdataModel)
    for ind in range(5):
        assert abs(float(mojoPredict[ind]['predict']) - modelPred[ind, 0]) < 1e-06, 'model predict {1} and mojo predict {0} differs too much'.format(float(mojoPredict[0]['predict']), modelPred[ind, 0])
if __name__ == '__main__':
    pyunit_utils.standalone_test(test_setInvNumNA)
else:
    test_setInvNumNA()