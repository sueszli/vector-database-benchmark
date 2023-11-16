from __future__ import annotations
import math
from river import base, utils
from river.tree.mondrian.mondrian_tree import MondrianTree
from river.tree.mondrian.mondrian_tree_nodes import MondrianBranchClassifier, MondrianLeafClassifier, MondrianNodeClassifier

class MondrianTreeClassifier(MondrianTree, base.Classifier):
    """Mondrian Tree classifier.

    Parameters
    ----------
    step
        Step of the tree.
    use_aggregation
        Whether to use aggregation weighting techniques or not.
    dirichlet
        Dirichlet parameter of the problem.
    split_pure
        Whether the tree should split pure leafs during training or not.
    iteration
        Number iterations to do during training.
    seed
        Random seed for reproducibility.

    Notes
    -----
    The Mondrian Tree Classifier is a type of decision tree that bases splitting decisions over a
    Mondrian process.

    Examples
    --------
    >>> from river import datasets
    >>> from river import evaluate
    >>> from river import metrics
    >>> from river import tree

    >>> dataset = datasets.Bananas().take(500)

    >>> model = tree.mondrian.MondrianTreeClassifier(
    ...     step=0.1,
    ...     use_aggregation=True,
    ...     dirichlet=0.2,
    ...     seed=1
    ... )

    >>> metric = metrics.Accuracy()

    >>> evaluate.progressive_val_score(dataset, model, metric)
    Accuracy: 58.52%

    References
    ----------
    [^1]: Balaji Lakshminarayanan, Daniel M. Roy, Yee Whye Teh. Mondrian Forests: Efficient Online Random Forests.
        arXiv:1406.2673, pages 2-4

    """

    def __init__(self, step: float=0.1, use_aggregation: bool=True, dirichlet: float=0.5, split_pure: bool=False, iteration: int=0, seed: int | None=None):
        if False:
            print('Hello World!')
        super().__init__(step=step, loss='log', use_aggregation=use_aggregation, iteration=iteration, seed=seed)
        self.dirichlet = dirichlet
        self.split_pure = split_pure
        self._classes: set[base.typing.ClfTarget] = set()
        self._x: dict[base.typing.FeatureName, int | float]
        self._y: base.typing.ClfTarget
        self._root = MondrianLeafClassifier(None, 0.0, 0)

    @property
    def _is_initialized(self):
        if False:
            i = 10
            return i + 15
        'Check if the tree has learnt at least one sample'
        return len(self._classes) != 0

    def _score(self, node: MondrianNodeClassifier) -> float:
        if False:
            print('Hello World!')
        'Computes the score of the node regarding the current sample being proceeded\n\n        Parameters\n        ----------\n        node\n            Node to evaluate the score.\n\n        '
        return node.score(self._y, self.dirichlet, len(self._classes))

    def _predict(self, node: MondrianNodeClassifier) -> dict[base.typing.ClfTarget, float]:
        if False:
            return 10
        'Compute the predictions scores of the node regarding all the classes scores.\n\n        Parameters\n        ----------\n        node\n            Node to make predictions.\n\n        '
        return node.predict(self.dirichlet, self._classes, len(self._classes))

    def _loss(self, node: MondrianNodeClassifier) -> float:
        if False:
            while True:
                i = 10
        'Compute the loss for the given node regarding the current label\n\n        Parameters\n        ----------\n        node\n            Node to evaluate the loss.\n\n        '
        return node.loss(self._y, self.dirichlet, len(self._classes))

    def _update_weight(self, node: MondrianNodeClassifier) -> float:
        if False:
            return 10
        'Update the weight of the node regarding the current label with the tree parameters.\n\n        Parameters\n        ----------\n        node\n            Node to update the weight.\n\n        '
        return node.update_weight(self._y, self.dirichlet, self.use_aggregation, self.step, len(self._classes))

    def _update_count(self, node: MondrianNodeClassifier):
        if False:
            print('Hello World!')
        'Update the count of labels with the current class `_y` being\n        treated (not to use twice for one sample added).\n\n        Parameters\n        ----------\n        node\n            Target node.\n\n        '
        node.update_count(self._y)

    def _update_downwards(self, x, y: base.typing.ClfTarget, node: MondrianNodeClassifier, do_weight_update: bool):
        if False:
            while True:
                i = 10
        'Update the node when running a downward procedure updating the tree.\n\n        Parameters\n        ----------\n        node\n            Target node.\n        do_weight_update\n            Whether we should update the weights or not.\n\n        '
        return node.update_downwards(x, y, self.dirichlet, self.use_aggregation, self.step, do_weight_update, len(self._classes))

    def _compute_split_time(self, y: base.typing.ClfTarget, node: MondrianLeafClassifier | MondrianBranchClassifier, extensions_sum: float) -> float:
        if False:
            while True:
                i = 10
        'Compute the spit time of the given node.\n\n        Parameters\n        ----------\n        node\n            Target node.\n\n        '
        if not self.split_pure and node.is_dirac(y):
            return 0.0
        if extensions_sum > 0:
            T = utils.random.exponential(1 / extensions_sum, rng=self._rng)
            time = node.time
            split_time = time + T
            if isinstance(node, MondrianLeafClassifier):
                return split_time
            (left, _) = node.children
            child_time = left.time
            if split_time < child_time:
                return split_time
        return 0.0

    def _split(self, node: MondrianLeafClassifier | MondrianBranchClassifier, split_time: float, threshold: float, feature: base.typing.FeatureName, is_right_extension: bool) -> MondrianBranchClassifier:
        if False:
            while True:
                i = 10
        'Split the given node and set the split time, threshold, etc., to the node.\n\n        Parameters\n        ----------\n        node\n            Target node.\n        split_time\n            Split time of the node in the Mondrian process.\n        threshold\n            Threshold of acceptance of the node.\n        feature\n            Feature of the node.\n        is_right_extension\n            Should we extend the tree in the right or left direction.\n\n        '
        new_depth = node.depth + 1
        left: MondrianLeafClassifier | MondrianBranchClassifier
        right: MondrianLeafClassifier | MondrianBranchClassifier
        if isinstance(node, MondrianBranchClassifier):
            (old_left, old_right) = node.children
            if is_right_extension:
                left = MondrianBranchClassifier(node, split_time, new_depth, node.feature, node.threshold)
                right = MondrianLeafClassifier(node, split_time, new_depth)
                left.replant(node)
                old_left.parent = left
                old_right.parent = left
                left.children = (old_left, old_right)
            else:
                right = MondrianBranchClassifier(node, split_time, new_depth, node.feature, node.threshold)
                left = MondrianLeafClassifier(node, split_time, new_depth)
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
        branch = MondrianBranchClassifier(node.parent, node.time, node.depth, feature, threshold)
        left = MondrianLeafClassifier(branch, split_time, new_depth)
        right = MondrianLeafClassifier(branch, split_time, new_depth)
        branch.children = (left, right)
        branch.replant(node, True)
        if is_right_extension:
            left.replant(node)
        else:
            right.replant(node)
        del node
        return branch

    def _go_downwards(self, x, y):
        if False:
            return 10
        'Update the tree (downward procedure).'
        current_node = self._root
        if self.iteration == 0:
            self._update_downwards(x, y, current_node, False)
            return current_node
        else:
            branch_no = None
            while True:
                (extensions_sum, extensions) = current_node.range_extension(x)
                split_time = self._compute_split_time(y, current_node, extensions_sum)
                if split_time > 0:
                    intensities = utils.norm.normalize_values_in_dict(extensions, inplace=False)
                    candidates = sorted(list(x.keys()))
                    feature = self._rng.choices(candidates, [intensities[c] for c in candidates], k=1)[0]
                    x_f = x[feature]
                    (range_min, range_max) = current_node.range(feature)
                    is_right_extension = x_f > range_max
                    if is_right_extension:
                        threshold = self._rng.uniform(range_max, x_f)
                    else:
                        threshold = self._rng.uniform(x_f, range_min)
                    was_leaf = isinstance(current_node, MondrianLeafClassifier)
                    current_node = self._split(current_node, split_time, threshold, feature, is_right_extension)
                    if current_node.parent is None:
                        self._root = current_node
                    elif was_leaf:
                        parent = current_node.parent
                        if branch_no == 0:
                            parent.children = (current_node, parent.children[1])
                        else:
                            parent.children = (parent.children[0], current_node)
                    self._update_downwards(x, y, current_node, True)
                    (left, right) = current_node.children
                    if is_right_extension:
                        current_node = right
                    else:
                        current_node = left
                    leaf = current_node
                    self._update_downwards(x, y, leaf, False)
                    return leaf
                else:
                    self._update_downwards(x, y, current_node, True)
                    if isinstance(current_node, MondrianLeafClassifier):
                        return current_node
                    else:
                        try:
                            branch_no = current_node.branch_no(x)
                            current_node = current_node.children[branch_no]
                        except KeyError:
                            (branch_no, current_node) = current_node.most_common_path()

    def _go_upwards(self, leaf: MondrianLeafClassifier):
        if False:
            return 10
        'Update the tree (upwards procedure).\n\n        Parameters\n        ----------\n        leaf\n            Leaf to start from when going upward.\n\n        '
        current_node = leaf
        if self.iteration >= 1:
            while True:
                current_node.update_weight_tree()
                if current_node.parent is None:
                    break
                current_node = current_node.parent

    @property
    def _multiclass(self):
        if False:
            while True:
                i = 10
        return True

    def learn_one(self, x, y):
        if False:
            print('Hello World!')
        self._classes.add(y)
        leaf = self._go_downwards(x, y)
        if self.use_aggregation:
            self._go_upwards(leaf)
        self.iteration += 1
        return self

    def predict_proba_one(self, x):
        if False:
            while True:
                i = 10
        'Predict the probability of the samples.\n\n        Parameters\n        ----------\n        x\n            Feature vector.\n\n        '
        if not self._is_initialized:
            return {}
        scores = {c: 0.0 for c in self._classes}
        leaf = self._root.traverse(x, until_leaf=True) if isinstance(self._root, MondrianBranchClassifier) else self._root
        if not self.use_aggregation:
            return self._predict(leaf)
        current = leaf
        while True:
            if isinstance(current, MondrianLeafClassifier):
                scores = self._predict(current)
            else:
                weight = current.weight
                log_weight_tree = current.log_weight_tree
                w = math.exp(weight - log_weight_tree)
                pred_new = self._predict(current)
                for c in self._classes:
                    scores[c] = 0.5 * w * pred_new[c] + (1 - 0.5 * w) * scores[c]
            if current.parent is None:
                break
            current = current.parent
        return utils.norm.normalize_values_in_dict(scores, inplace=False)