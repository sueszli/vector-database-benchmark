from past.utils import old_div
import sys
sys.path.insert(1, '../../../')
import h2o
from tests import pyunit_utils
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
import pandas as pd
import numpy as np

def interactions_GLM_Binomial():
    if False:
        while True:
            i = 10
    pd_df = pd.DataFrame(np.array([[0.1, 0.2, 0.3, 0.15, 0.25, 0.35, 0.12, 0.22, 0.32, 0.2, 0.3, 0.15, 0.05], ['a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'c', 'a', 'a', 'a', 'b'], ['Red', 'Blue', 'Green', 'Red', 'Blue', 'Green', 'Red', 'Blue', 'Green', 'Blue', 'Green', 'Red', 'Blue']]).T, columns=['label', 'categorical_feat', 'categorical_feat2'])
    h2o_df = h2o.H2OFrame(pd_df, na_strings=['UNKNOWN'])
    interaction_pairs = ['categorical_feat', 'categorical_feat2']
    model0 = H2OGeneralizedLinearEstimator(family='Gaussian', Lambda=0, interactions=interaction_pairs)
    model0.train(x=['categorical_feat', 'categorical_feat2'], y='label', training_frame=h2o_df)
    model1 = H2OGeneralizedLinearEstimator(family='Gaussian', Lambda=0.001, interactions=interaction_pairs)
    model1.train(x=['categorical_feat', 'categorical_feat2'], y='label', training_frame=h2o_df)
    model0CoeffLen = 4 + 2 + 2 + 1
    model1CoeffLen = 9 + 3 + 3 + 1
    assert len(model0.coef()) == model0CoeffLen, 'Lambda=0, Expected coefficient length: {0}, Actual: {1}'.format(model0CoeffLen, len(model0.coef()))
    assert len(model1.coef()) == model1CoeffLen, 'Lambda=0.001, Expected coefficient length: {0}, Actual: {1}'.format(model1CoeffLen, len(model1.coef()))
if __name__ == '__main__':
    pyunit_utils.standalone_test(interactions_GLM_Binomial)
else:
    interactions_GLM_Binomial()