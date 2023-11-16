import sys, os
sys.path.insert(1, os.path.join('..', '..', '..'))
import h2o
from tests import pyunit_utils
from h2o.estimators.deeplearning import H2ODeepLearningEstimator

def checkpoint_new_category_in_predictor():
    if False:
        print('Hello World!')
    sv1 = h2o.upload_file(pyunit_utils.locate('smalldata/iris/setosa_versicolor.csv'))
    sv2 = h2o.upload_file(pyunit_utils.locate('smalldata/iris/setosa_versicolor.csv'))
    vir = h2o.upload_file(pyunit_utils.locate('smalldata/iris/virginica.csv'))
    print('checkpoint_new_category_in_predictor-1')
    m1 = H2ODeepLearningEstimator(epochs=100)
    m1.train(x=[0, 1, 2, 4], y=3, training_frame=sv1)
    m2 = H2ODeepLearningEstimator(epochs=200, checkpoint=m1.model_id)
    m2.train(x=[0, 1, 2, 4], y=3, training_frame=sv2)
    print('checkpoint_new_category_in_predictor-2')
    try:
        m3 = H2ODeepLearningEstimator(epochs=200, checkpoint=m1.model_id)
        m3.train(x=[0, 1, 2, 4], y=3, training_frame=vir)
        assert False, 'Expected continued model-building to fail with new categories introduced in predictor'
    except EnvironmentError:
        pass
    print('checkpoint_new_category_in_predictor-3')
    predictions = m2.predict(vir)
    print('checkpoint_new_category_in_predictor-4')
if __name__ == '__main__':
    pyunit_utils.standalone_test(checkpoint_new_category_in_predictor)
else:
    checkpoint_new_category_in_predictor()