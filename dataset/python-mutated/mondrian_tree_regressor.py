from __future__ import annotations
import math
from river import base, utils
from river.tree.mondrian.mondrian_tree import MondrianTree
from river.tree.mondrian.mondrian_tree_nodes import MondrianBranchRegressor, MondrianLeafRegressor, MondrianNodeRegressor

class MondrianTreeRegressor(MondrianTree, base.Regressor):
    """Mondrian Tree Regressor.

    Parameters
    ----------
    step
        Step of the tree.
    use_aggregation
        Whether to use aggregation weighting techniques or not.
    iteration
        Number iterations to do during training.
    seed
        Random seed for reproducibility.

    Notes
    -----
    The Mondrian Tree Regressor is a type of decision tree that bases splitting decisions over a
    Mondrian process.

    References
    ----------
    [^1]: Balaji Lakshminarayanan, Daniel M. Roy, Yee Whye Teh. Mondrian Forests: Efficient Online Random Forests.
        arXiv:1406.2673, pages 2-4.

    """

    def __init__(self, step: float=0.1, use_aggregation: bool=True, iteration: int=0, seed: int=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(step=step, loss='least-squares', use_aggregation=use_aggregation, iteration=iteration, seed=seed)
        self.seed = seed
        self._x: dict[base.typing.FeatureName, int | float]
        self._y: base.typing.RegTarget
        self._root = MondrianLeafRegressor(None, 0.0, 0)

    def _is_initialized(self):
        if False:
            i = 10
            return i + 15
        'Check if the tree has learnt at least one sample'
        return self.iteration != 0

    def _predict(self, node: MondrianNodeRegressor) -> base.typing.RegTarget:
        if False:
            for i in range(10):
                print('nop')
        'Compute the prediction.\n\n        Parameters\n        ----------\n        node\n            Node to make predictions.\n\n        '
        return node.predict()

    def _loss(self, node: MondrianNodeRegressor) -> float:
        if False:
            while True:
                i = 10
        'Compute the loss for the given node regarding the current label.\n\n        Parameters\n        ----------\n        node\n            Node to evaluate the loss.\n\n        '
        return node.loss(self._y)

    def _update_weight(self, node: MondrianNodeRegressor) -> float:
        if False:
            print('Hello World!')
        'Update the weight of the node regarding the current label with the tree parameters.\n\n        Parameters\n        ----------\n        node\n            Node to update the weight.\n\n        '
        return node.update_weight(self._y, self.use_aggregation, self.step)

    def _update_downwards(self, node: MondrianNodeRegressor, do_update_weight):
        if False:
            for i in range(10):
                print('nop')
        'Update the node when running a downward procedure updating the tree.\n\n        Parameters\n        ----------\n        node\n            Target node.\n        do_update_weight\n            Whether we should update the weights or not.\n\n        '
        return node.update_downwards(self._x, self._y, self.use_aggregation, self.step, do_update_weight)

    def _compute_split_time(self, node: MondrianLeafRegressor | MondrianBranchRegressor, extensions_sum: float) -> float:
        if False:
            i = 10
            return i + 15
        'Computes the split time of the given node.\n\n        Parameters\n        ----------\n        node\n            Target node.\n\n        '
        if extensions_sum > 0:
            T = utils.random.exponential(1 / extensions_sum, rng=self._rng)
            time = node.time
            split_time = time + T
            if isinstance(node, MondrianLeafRegressor):
                return split_time
            (left, _) = node.children
            child_time = left.time
            if split_time < child_time:
                return split_time
        return 0.0

    def _split(self, node: MondrianLeafRegressor | MondrianBranchRegressor, split_time: float, threshold: float, feature: base.typing.FeatureName, is_right_extension: bool) -> MondrianBranchRegressor:
        if False:
            for i in range(10):
                print('nop')
        'Split the given node and attributes the split time, threshold, etc... to the node.\n\n        Parameters\n        ----------\n        node\n            Target node.\n        split_time\n            Split time of the node in the Mondrian process.\n        threshold\n            Threshold of acceptance of the node.\n        feature\n            Feature index of the node.\n        is_right_extension\n            Should we extend the tree in the right or left direction.\n\n        '
        new_depth = node.depth + 1
        left: MondrianLeafRegressor | MondrianBranchRegressor
        right: MondrianLeafRegressor | MondrianBranchRegressor
        if isinstance(node, MondrianBranchRegressor):
            (old_left, old_right) = node.children
            if is_right_extension:
                left = MondrianBranchRegressor(node, split_time, new_depth, node.feature, node.threshold)
                right = MondrianLeafRegressor(node, split_time, new_depth)
                left.replant(node)
                old_left.parent = left
                old_right.parent = left
                left.children = (old_left, old_right)
            else:
                right = MondrianBranchRegressor(node, split_time, new_depth, node.feature, node.threshold)
                left = MondrianLeafRegressor(node, split_time, new_depth)
                right.replant(node)
                old_left.parent = right
                old_right.parent = right
                right.children = (old_left, old_right)
            new_depth += 1
            old_left.update_depth(new_depth)
            old_right.update_depth(new_depth)
            node.feature = feature
            node.threshold = threshold
            node.children = (left, right)
            return node
        branch = MondrianBranchRegressor(node.parent, node.time, node.depth, feature, threshold)
        left = MondrianLeafRegressor(branch, split_time, new_depth)
        right = MondrianLeafRegressor(branch, split_time, new_depth)
        branch.children = (left, right)
        branch.replant(node, True)
        if is_right_extension:
            left.replant(node)
        else:
            right.replant(node)
        del node
        return branch

    def _go_downwards(self):
        if False:
            i = 10
            return i + 15
        'Update the tree (downward procedure).'
        current_node = self._root
        if self.iteration == 0:
            self._update_downwards(current_node, False)
            return current_node
        else:
            branch_no = None
            while True:
                (extensions_sum, extensions) = current_node.range_extension(self._x)
                split_time = self._compute_split_time(current_node, extensions_sum)
                if split_time > 0:
                    intensities = utils.norm.normalize_values_in_dict(extensions, inplace=False)
                    candidates = sorted(list(self._x.keys()))
                    feature = self._rng.choices(candidates, [intensities[c] for c in candidates], k=1)[0]
                    x_f = self._x[feature]
                    (range_min, range_max) = current_node.range(feature)
                    is_right_extension = x_f > range_max
                    if is_right_extension:
                        threshold = self._rng.uniform(range_max, x_f)
                    else:
                        threshold = self._rng.uniform(x_f, range_min)
                    was_leaf = isinstance(current_node, MondrianLeafRegressor)
                    current_node = self._split(current_node, split_time, threshold, feature, is_right_extension)
                    if current_node.parent is None:
                        self._root = current_node
                    elif was_leaf:
                        parent = current_node.parent
                        if branch_no == 0:
                            parent.children = (current_node, parent.children[1])
                        else:
                            parent.children = (parent.children[0], current_node)
                    self._update_downwards(current_node, True)
                    (left, right) = current_node.children
                    if is_right_extension:
                        current_node = right
                    else:
                        current_node = left
                    leaf = current_node
                    self._update_downwards(leaf, False)
                    return leaf
                else:
                    self._update_downwards(current_node, True)
                    if isinstance(current_node, MondrianLeafRegressor):
                        return current_node
                    else:
                        try:
                            branch_no = current_node.branch_no(self._x)
                            current_node = current_node.children[branch_no]
                        except KeyError:
                            (branch_no, current_node) = current_node.most_common_path()

    def _go_upwards(self, leaf: MondrianLeafRegressor):
        if False:
            print('Hello World!')
        'Update the tree (upwards procedure).\n\n        Parameters\n        ----------\n        leaf\n            Leaf to start from when going upward.\n\n        '
        current_node = leaf
        if self.iteration >= 1:
            while True:
                current_node.update_weight_tree()
                if current_node.parent is None:
                    break
                current_node = current_node.parent

    def learn_one(self, x, y):
        if False:
            return 10
        self._x = x
        self._y = y
        leaf = self._go_downwards()
        if self.use_aggregation:
            self._go_upwards(leaf)
        self.iteration += 1
        return self

    def predict_one(self, x):
        if False:
            i = 10
            return i + 15
        'Predict the label of the samples.\n\n        Parameters\n        ----------\n        x\n            Feature vector.\n\n        '
        if not self._is_initialized:
            return
        leaf = self._root.traverse(x, until_leaf=True) if isinstance(self._root, MondrianBranchRegressor) else self._root
        if not self.use_aggregation:
            return self._predict(leaf)
        current = leaf
        prediction = 0.0
        while True:
            if isinstance(current, MondrianLeafRegressor):
                prediction = self._predict(current)
            else:
                weight = current.weight
                log_weight_tree = current.log_weight_tree
                w = math.exp(weight - log_weight_tree)
                pred_new = self._predict(current)
                prediction = 0.5 * w * pred_new + (1 - 0.5 * w) * prediction
            if current.parent is None:
                break
            current = current.parent
        return prediction