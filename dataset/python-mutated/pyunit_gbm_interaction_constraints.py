from builtins import range
import sys
sys.path.insert(1, '../../../')
import h2o
from tests import pyunit_utils
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.tree import H2OTree

def test_interaction_constraints():
    if False:
        return 10
    prostate = h2o.import_file(path=pyunit_utils.locate('smalldata/logreg/prostate.csv'))
    prostate.describe()
    prostate[1] = prostate[1].asfactor()
    constraints = [['AGE', 'PSA'], ['GLEASON']]
    ntrees = 5
    prostate_gbm = H2OGradientBoostingEstimator(distribution='bernoulli', ntrees=ntrees, interaction_constraints=constraints, seed=42)
    prostate_gbm.train(x=list(range(2, 9)), y=1, training_frame=prostate)
    prostate_gbm.predict(prostate)
    importance = prostate_gbm.varimp(use_pandas=True)
    print(importance)
    assert importance['variable'][3] == 'RACE'
    assert importance['relative_importance'][3] == 0
    assert importance['variable'][4] == 'DPROS'
    assert importance['relative_importance'][4] == 0
    assert importance['variable'][5] == 'DCAPS'
    assert importance['relative_importance'][5] == 0
    assert importance['variable'][6] == 'VOL'
    assert importance['relative_importance'][6] == 0
    for i in range(ntrees):
        tree = H2OTree(model=prostate_gbm, tree_number=i)
        tree_features = set(filter(None, tree.features))
        print('iteration: ' + str(i))
        print(set(constraints[0]))
        print(set(constraints[1]))
        print(tree_features)
        assert tree_features.issubset(set(constraints[0])) or tree_features.issubset(set(constraints[1]))
if __name__ == '__main__':
    pyunit_utils.standalone_test(test_interaction_constraints)
else:
    test_interaction_constraints()