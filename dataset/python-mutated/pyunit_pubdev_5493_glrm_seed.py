import sys
sys.path.insert(1, '../../../')
import h2o
from tests import pyunit_utils
from h2o.estimators.glrm import H2OGeneralizedLowRankEstimator
'\nPUBDEV-5493: GLRM return different results regardless of seed setting.\n'

def test_glrm_seeds():
    if False:
        i = 10
        return i + 15
    print('Importing iris_wheader.csv data...')
    irisH2O = h2o.upload_file(pyunit_utils.locate('smalldata/iris/iris_wheader.csv'))
    irisH2O.describe()
    initMethods = ['random', 'svd', 'plus_plus', 'user']
    seeds = [123456789, 987654321]
    for initM in initMethods:
        glrm_h2o_seed0 = setupTrainModel(initM, seeds[0])
        predict_seed0 = predGLRM(irisH2O, glrm_h2o_seed0)
        glrm_h2o_seed0Same = setupTrainModel(initM, seeds[0])
        predict_seed0Same = predGLRM(irisH2O, glrm_h2o_seed0Same)
        pyunit_utils.compare_frames_local(predict_seed0[0:4], predict_seed0Same[0:4], prob=1.0)
        glrm_h2o_seed1 = setupTrainModel(initM, seeds[1])
        predict_seed1 = predGLRM(irisH2O, glrm_h2o_seed1)
        assert not pyunit_utils.compare_frames_local(predict_seed0[0:4], predict_seed1[0:4], prob=1.0, returnResult=True), 'GLRM return same results with different random seed.'

def setupTrainModel(initM, seed):
    if False:
        i = 10
        return i + 15
    rank = 3
    gx = 0.25
    gy = 0.25
    trans = 'STANDARDIZE'
    return H2OGeneralizedLowRankEstimator(k=rank, loss='Quadratic', gamma_x=gx, gamma_y=gy, transform=trans, init=initM, seed=seed)

def predGLRM(dataset, model):
    if False:
        for i in range(10):
            print('nop')
    '\n  Simple method to train GLRM model and return prediction result.\n\n  :param dataset: dataset to be scored and trained on\n  :param model: glrm model to be trained\n  :return: reconstructed dataframe.\n  '
    model.train(x=dataset.names, training_frame=dataset)
    return model.predict(dataset)
if __name__ == '__main__':
    pyunit_utils.standalone_test(test_glrm_seeds)
else:
    test_glrm_seeds()