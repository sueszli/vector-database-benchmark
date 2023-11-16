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
    print('******* Test interaction pairs')
    pd_df_NA = pd.DataFrame(np.array([[1, 0, 1, 0, 1, 0], [1, 2, 4.2 / 2.2, 4, 3, 1], [2, 3, float('NaN'), 1, 2, 3], ['a', 'a', 'a', 'b', 'a', 'b'], ['Foo', 'UNKNOWN', 'Foo', 'Foo', 'Foo', 'Bar']]).T, columns=['label', 'numerical_feat', 'numerical_feat2', 'categorical_feat', 'categorical_feat2'])
    h2o_df_NA = h2o.H2OFrame(pd_df_NA, na_strings=['UNKNOWN'])
    pd_df = pd.DataFrame(np.array([[1, 0, 1, 0, 1, 0], [1, 2, 4.2 / 2.2, 4, 3, 1], [2, 3, 2.2, 1, 2, 3], ['a', 'a', 'a', 'b', 'a', 'b'], ['Foo', 'Foo', 'Foo', 'Foo', 'Foo', 'Bar']]).T, columns=['label', 'numerical_feat', 'numerical_feat2', 'categorical_feat', 'categorical_feat2'])
    h2o_df = h2o.H2OFrame(pd_df, na_strings=['UNKNOWN'])
    interaction_pairs = [('numerical_feat', 'numerical_feat2'), ('numerical_feat', 'categorical_feat2'), ('categorical_feat', 'categorical_feat2')]
    xcols = ['numerical_feat', 'numerical_feat2', 'categorical_feat', 'categorical_feat2']
    modelNA = H2OGeneralizedLinearEstimator(family='Binomial', alpha=0, lambda_search=False, interaction_pairs=interaction_pairs, standardize=False)
    modelNA.train(x=xcols, y='label', training_frame=h2o_df_NA)
    model = H2OGeneralizedLinearEstimator(family='Binomial', alpha=0, lambda_search=False, interaction_pairs=interaction_pairs, standardize=False)
    model.train(x=xcols, y='label', training_frame=h2o_df)
    assert_arrays_equal_NA(modelNA._model_json['output']['coefficients_table'].cell_values, model._model_json['output']['coefficients_table'].cell_values)
    print('******* Test interaction with num by num')
    pd_df_num_num_NA = pd.DataFrame(np.array([[1, 0, 1, 0], [1, 2, 2, 4], [2, 3, float('NaN'), 1]]).T, columns=['label', 'numerical_feat', 'numerical_feat2'])
    pd_df_num_num = pd.DataFrame(np.array([[1, 0, 1, 0], [1, 2, 2, 4], [2, 3, 2, 1]]).T, columns=['label', 'numerical_feat', 'numerical_feat2'])
    performOneTest(pd_df_num_num_NA, pd_df_num_num, interactionColumn=['numerical_feat', 'numerical_feat2'], xcols=['numerical_feat', 'numerical_feat2'], standard=False)
    print('******* Test interaction with enum by enum')
    pd_df_cat_cat_NA = pd.DataFrame(np.array([[1, 0, 1, 0], ['a', 'a', 'b', 'b'], ['Foo', 'UNKNOWN', 'Foo', 'Bar']]).T, columns=['label', 'categorical_feat', 'categorical_feat2'])
    pd_df_cat_cat = pd.DataFrame(np.array([[1, 0, 1, 0], ['a', 'a', 'b', 'b'], ['Foo', 'Foo', 'Foo', 'Bar']]).T, columns=['label', 'categorical_feat', 'categorical_feat2'])
    performOneTest(pd_df_cat_cat_NA, pd_df_cat_cat, interactionColumn=['categorical_feat', 'categorical_feat2'], xcols=['categorical_feat', 'categorical_feat2'])
    print('******* Test interaction with enum by num')
    pd_df_cat_num_NA = pd.DataFrame(np.array([[1, 0, 1, 0], [1, 2, 3, 4], ['Foo', 'UNKNOWN', 'Foo', 'Bar']]).T, columns=['label', 'numerical_feat', 'categorical_feat'])
    pd_df_cat_num = pd.DataFrame(np.array([[1, 0, 1, 0], [1, 2, 3, 4], ['Foo', 'Foo', 'Foo', 'Bar']]).T, columns=['label', 'numerical_feat', 'categorical_feat'])
    performOneTest(pd_df_cat_num_NA, pd_df_cat_num, interactionColumn=['numerical_feat', 'categorical_feat'], xcols=['numerical_feat', 'categorical_feat'])

def performOneTest(frameWithNA, frameWithoutNA, interactionColumn, xcols, standard=True):
    if False:
        for i in range(10):
            print('nop')
    h2o_df_NA = h2o.H2OFrame(frameWithNA, na_strings=['UNKNOWN'])
    h2o_df_NA_Valid = h2o.H2OFrame(frameWithNA, na_strings=['UNKNOWN'])
    h2o_df = h2o.H2OFrame(frameWithoutNA, na_strings=['UNKNOWN'])
    h2o_df_valid = h2o.H2OFrame(frameWithoutNA, na_strings=['UNKNOWN'])
    modelNA = H2OGeneralizedLinearEstimator(family='Binomial', alpha=0, lambda_search=False, interactions=interactionColumn, standardize=standard)
    modelNA.train(x=xcols, y='label', training_frame=h2o_df_NA, validation_frame=h2o_df_NA_Valid)
    model = H2OGeneralizedLinearEstimator(family='Binomial', alpha=0, lambda_search=False, interactions=interactionColumn, standardize=standard)
    model.train(x=xcols, y='label', training_frame=h2o_df, validation_frame=h2o_df_valid)
    coef_m_NA = modelNA._model_json['output']['coefficients_table']
    coef_m = model._model_json['output']['coefficients_table']
    if not len(coef_m_NA.cell_values) == len(coef_m.cell_values):
        assert_arrays_equal_NA(coef_m_NA.cell_values, coef_m.cell_values)
    else:
        pyunit_utils.equal_2d_tables(coef_m_NA.cell_values, coef_m.cell_values)

def assert_arrays_equal_NA(coef_m_NA_dict, coef_m):
    if False:
        print('Hello World!')
    coefNAIndex = 0
    for index in range(len(coef_m)):
        if not coef_m_NA_dict[coefNAIndex][0] == coef_m[index][0]:
            coefNAIndex = coefNAIndex + 1
        assert abs(coef_m_NA_dict[coefNAIndex][1] - coef_m[index][1]) < 1e-06, 'Expected: {0}, Actual: {1}'.format(coef_m_NA_dict[coefNAIndex][1], coef_m[index][1])
        coefNAIndex = coefNAIndex + 1
if __name__ == '__main__':
    pyunit_utils.standalone_test(interactions_GLM_Binomial)
else:
    interactions_GLM_Binomial()