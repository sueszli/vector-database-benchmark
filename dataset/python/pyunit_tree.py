import h2o

from h2o.tree import H2OTree
from h2o.estimators import H2OGradientBoostingEstimator
from h2o.estimators import H2ORandomForestEstimator
from h2o.estimators import H2OIsolationForestEstimator
from tests import pyunit_utils


def check_tree(tree, tree_number, tree_class = None):
    print(tree)
    print(tree._root_node)

    assert tree is not None
    assert len(tree) > 0
    assert tree._tree_number == tree_number
    assert tree._tree_class == tree_class

    assert tree.root_node is not None
    assert tree.left_children is not None
    assert tree.right_children is not None
    assert tree.thresholds is not None
    assert tree.nas is not None
    assert tree.descriptions is not None
    assert tree.node_ids is not None
    assert tree.model_id is not None
    assert tree.levels is not None
    assert tree.root_node.na_direction is not None
    assert tree.root_node.id is not None


def tree_test():

    # GBM
    airlines = h2o.import_file(path=pyunit_utils.locate("smalldata/testng/airlines_train.csv"))
    gbm = H2OGradientBoostingEstimator(ntrees = 1)
    gbm.train(x = ["Origin", "Dest"], y = "IsDepDelayed", training_frame=airlines)

    tree = H2OTree(gbm, 0, "NO") # Indexing from 0 in Python. There is exactly one tree built
    check_tree(tree, 0, "NO")
    assert tree.root_node.left_levels is not None#Only categoricals in the model, guaranteed to have categorical split
    assert tree.root_node.right_levels is not None #Only categoricals in the model, guaranteed to have categorical split
    assert tree.left_cat_split is not None and tree.right_cat_split is not None
    assert len(tree.left_cat_split) == len(tree.right_cat_split)

    # There are categorical splits only, check none of the cat splits is None
    for i in range(0, len(tree.left_cat_split)):
        if(tree.left_children[i] == -1 and tree.right_children[i] == -1): # Except leaf nodes, those should be None
            assert tree.left_cat_split[i] is None
            assert tree.right_cat_split[i] is None
        else:
            assert tree.left_cat_split[i] is not None
            assert tree.right_cat_split[i] is not None

    # DRF
    cars = h2o.import_file(path=pyunit_utils.locate("smalldata/junit/cars_nice_header.csv"))
    drf = H2ORandomForestEstimator(ntrees=2)
    drf.train(x = ["power", "acceleration"], y="cylinders", training_frame=cars)

    drf_tree = H2OTree(drf, 1, None)
    check_tree(drf_tree, 1)

    # ISOFOR
    ecg_discord = h2o.import_file(pyunit_utils.locate("smalldata/anomaly/ecg_discord_train.csv"))
    isofor = H2OIsolationForestEstimator(ntrees=3, seed=12, sample_size=5)
    isofor.train(training_frame=ecg_discord)

    if_tree = H2OTree(isofor, 2)

    # There are no categoricall splits, check none of the cat splits is None
    for i in range(0, len(if_tree.node_ids)):
        assert if_tree.left_cat_split[i] is None
        assert if_tree.right_cat_split[i] is None
        if(if_tree.left_children[i] == -1 and tree.right_children[i] == -1): # Leaf nodes don't have split thresholds
            assert if_tree.thresholds is None
        else: # All others nodes should have split thresholds
            assert if_tree.thresholds[i] is not None
            assert if_tree.thresholds[i] is not None
    check_tree(if_tree, 2)


if __name__ == "__main__":
    pyunit_utils.standalone_test(tree_test)
else:
    tree_test()
