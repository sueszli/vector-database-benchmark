from builtins import range
import sys
sys.path.insert(1, '../../../')
import h2o
from tests import pyunit_utils, assert_equals, compare_frames, assert_not_equal
from h2o.estimators import H2OGradientBoostingEstimator
from h2o.tree import H2OTree

def assert_list_equals(expected, actual, delta=0):
    if False:
        print('Hello World!')
    assert_equals(len(expected), len(actual))
    for i in range(0, len(expected)):
        assert_equals(expected[i], actual[i], delta=delta)

def models_are_equal(model_1, model_2, pred_1, pred_2):
    if False:
        return 10
    assert_equals(True, compare_frames(pred_1, pred_2, pred_1.nrows))
    for tree_id in range(model_1.ntrees):
        for output_class in model_1._model_json['output']['domains'][-1]:
            tree = H2OTree(model=model_1, tree_number=tree_id, tree_class=output_class)
            tree2 = H2OTree(model=model_2, tree_number=tree_id, tree_class=output_class)
            assert_list_equals(tree.predictions, tree2.predictions)
            assert_list_equals(tree.thresholds, tree2.thresholds, delta=1e-50)
            assert_list_equals(tree.decision_paths, tree2.decision_paths)

def gbm_check_variable_importance_and_model(data, target):
    if False:
        return 10
    fr = h2o.import_file(pyunit_utils.locate(data))
    fr[target] = fr[target].asfactor()
    variable_importance_same = []
    ntrees = 50
    for i in range(5):
        model_1 = H2OGradientBoostingEstimator(ntrees=ntrees, sample_rate=1, seed=1234)
        model_1.train(y=target, training_frame=fr)
        pred_1 = model_1.predict(fr)
        model_2 = H2OGradientBoostingEstimator(ntrees=ntrees, sample_rate=1, seed=1234)
        model_2.train(y=target, training_frame=fr)
        pred_2 = model_2.predict(fr)
        relative_importance1 = model_1.varimp()
        relative_importance2 = model_2.varimp()
        variable_importance_same.append(relative_importance1 == relative_importance2)
        print(relative_importance1)
        print(relative_importance2)
        models_are_equal(model_1, model_2, pred_1, pred_2)
    return variable_importance_same

def gbm_reproducibility_variable_importance_different_but_same_model_small():
    if False:
        print('Hello World!')
    variable_importance_same = gbm_check_variable_importance_and_model('smalldata/prostate/prostate.csv', 'RACE')
    print(variable_importance_same)
    assert_equals(False, all(variable_importance_same))

def gbm_reproducibility_variable_importance_different_but_same_model_large():
    if False:
        print('Hello World!')
    variable_importance_same = gbm_check_variable_importance_and_model('bigdata/laptop/covtype/covtype.full.csv', 'Cover_Type')
    print(variable_importance_same)
    assert_equals(False, all(variable_importance_same))
if __name__ == '__main__':
    pyunit_utils.run_tests([gbm_reproducibility_variable_importance_different_but_same_model_small, gbm_reproducibility_variable_importance_different_but_same_model_large])
else:
    pyunit_utils.run_tests([gbm_reproducibility_variable_importance_different_but_same_model_small, gbm_reproducibility_variable_importance_different_but_same_model_large])