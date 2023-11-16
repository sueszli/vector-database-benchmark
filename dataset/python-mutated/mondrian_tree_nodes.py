from __future__ import annotations
import collections
import math
from river import base, stats
from river.tree.base import Branch, Leaf
from river.utils.math import log_sum_2_exp

class MondrianLeaf(Leaf):
    """Prototype class for all types of nodes in a Mondrian Tree.

    Parameters
    ----------
    parent
        Parent Node.
    time
        Split time of the node for Mondrian process.
    depth
        Depth of the leaf.

    """

    def __init__(self, parent, time, depth):
        if False:
            print('Hello World!')
        super().__init__()
        self.parent = parent
        self.time = time
        self.depth = depth

    @property
    def __repr__(self):
        if False:
            return 10
        return f'MondrianLeaf : {self.parent}, {self.time}, {self.depth}'

class MondrianBranch(Branch):

    def __init__(self, parent, time, depth, feature, threshold, *children):
        if False:
            while True:
                i = 10
        super().__init__(*children)
        self.parent = parent
        self.time = time
        self.depth = depth
        self.feature = feature
        self.threshold = threshold

    def branch_no(self, x) -> int:
        if False:
            while True:
                i = 10
        if x[self.feature] <= self.threshold:
            return 0
        return 1

    def next(self, x):
        if False:
            print('Hello World!')
        return self.children[self.branch_no(x)]

    def most_common_path(self):
        if False:
            print('Hello World!')
        (left, right) = self.children
        if left.weight < right.weight:
            return (1, right)
        return (0, left)

    def repr_split(self):
        if False:
            i = 10
            return i + 15
        return f'{self.feature} ≤ {self.threshold}'

class MondrianNode(base.Base):
    """Representation of a node within a Mondrian tree"""

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(*args, **kwargs)
        self.memory_range_min = collections.defaultdict(int)
        self.memory_range_max = collections.defaultdict(int)
        self.weight = 0.0
        self.log_weight_tree = 0.0

    def update_depth(self, depth):
        if False:
            while True:
                i = 10
        'Update the depth of the current node with the given depth.\n\n        Parameters\n        ----------\n        depth\n            Depth of the node.\n\n        '
        self.depth = depth
        if isinstance(self, MondrianLeaf):
            return
        depth += 1
        (left, right) = self.children
        left.update_depth(depth)
        right.update_depth(depth)

    def update_weight_tree(self):
        if False:
            print('Hello World!')
        'Update the weight of the node in the tree.'
        if isinstance(self, MondrianLeaf):
            self.log_weight_tree = self.weight
        else:
            (left, right) = self.children
            self.log_weight_tree = log_sum_2_exp(self.weight, left.log_weight_tree + right.log_weight_tree)

    def range(self, feature) -> tuple[float, float]:
        if False:
            return 10
        'Output the known range of the node regarding the j-th feature.\n\n        Parameters\n        ----------\n        feature\n            Feature for which you want to know the range.\n\n        '
        return (self.memory_range_min[feature], self.memory_range_max[feature])

    def range_extension(self, x) -> tuple[float, dict[base.typing.ClfTarget, float]]:
        if False:
            while True:
                i = 10
        'Compute the range extension of the node for the given sample.\n\n        Parameters\n        ----------\n        x\n            Sample to deal with.\n\n        '
        extensions: dict[base.typing.ClfTarget, float] = {}
        extensions_sum = 0.0
        for feature in x:
            x_f = x[feature]
            (feature_min_j, feature_max_j) = self.range(feature)
            if x_f < feature_min_j:
                diff = feature_min_j - x_f
            elif x_f > feature_max_j:
                diff = x_f - feature_max_j
            else:
                diff = 0
            extensions[feature] = diff
            extensions_sum += diff
        return (extensions_sum, extensions)

class MondrianNodeClassifier(MondrianNode):

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(*args, **kwargs)
        self.n_samples = 0
        self.counts = collections.defaultdict(int)

    def replant(self, leaf: MondrianNodeClassifier, copy_all: bool=False):
        if False:
            i = 10
            return i + 15
        'Transfer information from a leaf to a new branch.'
        self.weight = leaf.weight
        self.log_weight_tree = leaf.log_weight_tree
        if copy_all:
            self.memory_range_min = leaf.memory_range_min
            self.memory_range_max = leaf.memory_range_max
            self.n_samples = leaf.n_samples

    def score(self, y: base.typing.ClfTarget, dirichlet: float, n_classes: int) -> float:
        if False:
            print('Hello World!')
        'Compute the score of the node.\n\n        Parameters\n        ----------\n        y\n            Class for which we want the score.\n        dirichlet\n            Dirichlet parameter of the tree.\n        n_classes\n            The total number of classes seen so far.\n\n        Notes\n        -----\n        This uses Jeffreys prior with Dirichlet parameter for smoothing.\n\n        '
        count = self.counts[y]
        return (count + dirichlet) / (self.n_samples + dirichlet * n_classes)

    def predict(self, dirichlet: float, classes: set, n_classes: int) -> dict[base.typing.ClfTarget, float]:
        if False:
            print('Hello World!')
        'Predict the scores of all classes and output a `scores` dictionary\n        with the new values.\n\n        Parameters\n        ----------\n        dirichlet\n            Dirichlet parameter of the tree.\n        classes\n            The set of classes seen so far\n        n_classes\n            The total number of classes of the problem.\n\n        '
        scores = {}
        for c in classes:
            scores[c] = self.score(c, dirichlet, n_classes)
        return scores

    def loss(self, y: base.typing.ClfTarget, dirichlet: float, n_classes: int) -> float:
        if False:
            i = 10
            return i + 15
        'Compute the loss of the node.\n\n        Parameters\n        ----------\n        y\n            A given class of the problem.\n        dirichlet\n            Dirichlet parameter of the problem.\n        n_classes\n            The total number of classes of the problem.\n\n        '
        sc = self.score(y, dirichlet, n_classes)
        return -math.log(sc)

    def update_weight(self, y: base.typing.ClfTarget, dirichlet: float, use_aggregation: bool, step: float, n_classes: int) -> float:
        if False:
            return 10
        'Update the weight of the node given a class and the method used.\n\n        Parameters\n        ----------\n        y\n            Class of a given sample.\n        dirichlet\n            Dirichlet parameter of the tree.\n        use_aggregation\n            Whether to use aggregation or not during computation (given by the tree).\n        step\n            Step parameter of the tree.\n        n_classes\n            The total number of classes of the problem.\n\n        '
        loss_t = self.loss(y, dirichlet, n_classes)
        if use_aggregation:
            self.weight -= step * loss_t
        return loss_t

    def update_count(self, y):
        if False:
            print('Hello World!')
        'Update the amount of samples that belong to a class in the node\n        (not to use twice if you add one sample).\n\n        Parameters\n        ----------\n        y\n            Class of a given sample.\n\n        '
        self.counts[y] += 1

    def is_dirac(self, y: base.typing.ClfTarget) -> bool:
        if False:
            print('Hello World!')
        'Check whether the node follows a dirac distribution regarding the given\n        class, i.e., if the node is pure regarding the given class.\n\n        Parameters\n        ----------\n        y\n            Class of a given sample.\n\n        '
        return self.n_samples == self.counts[y]

    def update_downwards(self, x, y: base.typing.ClfTarget, dirichlet: float, use_aggregation: bool, step: float, do_update_weight: bool, n_classes: int):
        if False:
            i = 10
            return i + 15
        'Update the node when running a downward procedure updating the tree.\n\n        Parameters\n        ----------\n        x\n            Sample to proceed.\n        y\n            Class of the sample x.\n        dirichlet\n            Dirichlet parameter of the tree.\n        use_aggregation\n            Should it use the aggregation or not\n        step\n            Step of the tree.\n        do_update_weight\n            Should we update the weights of the node as well.\n        n_classes\n            The total number of classes of the problem.\n\n        '
        if self.n_samples == 0:
            for feature in x:
                x_f = x[feature]
                self.memory_range_min[feature] = x_f
                self.memory_range_max[feature] = x_f
        else:
            for feature in x:
                x_f = x[feature]
                if x_f < self.memory_range_min[feature]:
                    self.memory_range_min[feature] = x_f
                if x_f > self.memory_range_max[feature]:
                    self.memory_range_max[feature] = x_f
        self.n_samples += 1
        if do_update_weight:
            self.update_weight(y, dirichlet, use_aggregation, step, n_classes)
        self.update_count(y)

class MondrianLeafClassifier(MondrianNodeClassifier, MondrianLeaf):
    """Mondrian Tree Classifier leaf node.

    Parameters
    ----------
    parent
        Parent node.
    time
        Split time of the node.
    depth
        The depth of the leaf.

    """

    def __init__(self, parent, time, depth):
        if False:
            print('Hello World!')
        super().__init__(parent, time, depth)

class MondrianBranchClassifier(MondrianNodeClassifier, MondrianBranch):
    """Mondrian Tree Classifier branch node.

    Parameters
    ----------
    parent
        Parent node of the branch.
    time
        Split time characterizing the branch.
    depth
        Depth of the branch in the tree.
    feature
        Feature of the branch.
    threshold
        Acceptation threshold of the branch.
    *children
        Children nodes of the branch.

    """

    def __init__(self, parent, time, depth, feature, threshold, *children):
        if False:
            return 10
        super().__init__(parent, time, depth, feature, threshold, *children)

class MondrianNodeRegressor(MondrianNode):

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*args, **kwargs)
        self.n_samples = 0
        self.mean = stats.Mean()

    def replant(self, leaf: MondrianNodeRegressor, copy_all: bool=False):
        if False:
            return 10
        'Transfer information from a leaf to a new branch.'
        self.weight = leaf.weight
        self.log_weight_tree = leaf.log_weight_tree
        self.mean = leaf.mean
        if copy_all:
            self.memory_range_min = leaf.memory_range_min
            self.memory_range_max = leaf.memory_range_max
            self.n_samples = leaf.n_samples

    def predict(self) -> base.typing.RegTarget:
        if False:
            for i in range(10):
                print('nop')
        'Return the prediction of the node.'
        return self.mean.get()

    def loss(self, sample_value: base.typing.RegTarget) -> float:
        if False:
            while True:
                i = 10
        'Compute the loss of the node.\n\n        Parameters\n        ----------\n        sample_value\n            A given value.\n\n        '
        r = self.predict() - sample_value
        return r * r / 2

    def update_weight(self, sample_value: base.typing.RegTarget, use_aggregation: bool, step: float) -> float:
        if False:
            i = 10
            return i + 15
        'Update the weight of the node given a label and the method used.\n\n        Parameters\n        ----------\n        sample_value\n            Label of a given sample.\n        use_aggregation\n            Whether to use aggregation or not during computation (given by the tree).\n        step\n            Step parameter of the tree.\n\n        '
        loss_t = self.loss(sample_value)
        if use_aggregation:
            self.weight -= step * loss_t
        return loss_t

    def update_downwards(self, x, sample_value: base.typing.RegTarget, use_aggregation: bool, step: float, do_update_weight: bool):
        if False:
            for i in range(10):
                print('nop')
        'Update the node when running a downward procedure updating the tree.\n\n        Parameters\n        ----------\n        x\n            Sample to proceed (as a list).\n        sample_value\n            Label of the sample x.\n        use_aggregation\n            Should it use the aggregation or not\n        step\n            Step of the tree.\n        do_update_weight\n            Should we update the weights of the node as well.\n\n        '
        if self.n_samples == 0:
            for feature in x:
                x_f = x[feature]
                self.memory_range_min[feature] = x_f
                self.memory_range_max[feature] = x_f
        else:
            for feature in x:
                x_f = x[feature]
                if x_f < self.memory_range_min[feature]:
                    self.memory_range_min[feature] = x_f
                if x_f > self.memory_range_max[feature]:
                    self.memory_range_max[feature] = x_f
        self.n_samples += 1
        if do_update_weight:
            self.update_weight(sample_value, use_aggregation, step)
        self.mean.update(sample_value)

class MondrianLeafRegressor(MondrianNodeRegressor, MondrianLeaf):
    """Mondrian Tree Regressor leaf node.

    Parameters
    ----------
    parent
        Parent node.
    time
        Split time of the node.
    depth
        The depth of the leaf.

    """

    def __init__(self, parent, time, depth):
        if False:
            print('Hello World!')
        super().__init__(parent, time, depth)

class MondrianBranchRegressor(MondrianNodeRegressor, MondrianBranch):
    """Mondrian Tree Regressor branch node.

    Parameters
    ----------
    parent
        Parent node of the branch.
    time
        Split time characterizing the branch.
    depth
        Depth of the branch in the tree.
    feature
        Feature of the branch.
    threshold
        Acceptation threshold of the branch.
    *children
        Children nodes of the branch.

    """

    def __init__(self, parent, time, depth, feature, threshold, *children):
        if False:
            while True:
                i = 10
        super().__init__(parent, time, depth, feature, threshold, *children)