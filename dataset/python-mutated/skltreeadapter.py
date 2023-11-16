"""Tree adapter class for sklearn trees."""
from collections import OrderedDict
import random
import numpy as np
from Orange.widgets.visualize.utils.tree.treeadapter import BaseTreeAdapter
from Orange.misc.cache import memoize_method
from Orange.preprocess.transformation import Indicator
from Orange.widgets.visualize.utils.tree.rules import DiscreteRule, ContinuousRule

class SklTreeAdapter(BaseTreeAdapter):
    """Sklear Tree Adapter.

    An abstraction on top of the scikit learn classification tree.

    Parameters
    ----------
    model : SklModel
        An sklearn tree model instance.

    """
    NO_CHILD = -1
    FEATURE_UNDEFINED = -2

    def __init__(self, model):
        if False:
            i = 10
            return i + 15
        super().__init__(model)
        self._tree = model.skl_model.tree_
        self._all_leaves = None

    @memoize_method(maxsize=1024)
    def weight(self, node):
        if False:
            i = 10
            return i + 15
        return self.num_samples(node) / self.num_samples(self.parent(node))

    def num_samples(self, node):
        if False:
            for i in range(10):
                print('nop')
        return self._tree.n_node_samples[node]

    @memoize_method(maxsize=1024)
    def parent(self, node):
        if False:
            while True:
                i = 10
        for children in (self._tree.children_left, self._tree.children_right):
            try:
                return (children == node).nonzero()[0][0]
            except IndexError:
                continue
        return self.ROOT_PARENT

    def has_children(self, node):
        if False:
            print('Hello World!')
        return self._tree.children_left[node] != self.NO_CHILD or self._tree.children_right[node] != self.NO_CHILD

    def children(self, node):
        if False:
            return 10
        if self.has_children(node):
            return (self.__left_child(node), self.__right_child(node))
        return ()

    def __left_child(self, node):
        if False:
            print('Hello World!')
        return self._tree.children_left[node]

    def __right_child(self, node):
        if False:
            print('Hello World!')
        return self._tree.children_right[node]

    def reverse_children(self, node):
        if False:
            i = 10
            return i + 15
        (self._tree.children_left[node], self._tree.children_right[node]) = (self._tree.children_right[node], self._tree.children_left[node])

    def shuffle_children(self):
        if False:
            i = 10
            return i + 15
        for i in range(self.num_nodes):
            if random.randrange(2) == 0:
                self.reverse_children(i)

    @memoize_method(maxsize=1024)
    def get_distribution(self, node):
        if False:
            for i in range(10):
                print('nop')
        value = self._tree.value[node]
        if value.shape[1] == 1:
            var = np.var(self.get_instances_in_nodes(node).Y)
            variances = np.array([var * np.ones(value.shape[0])]).T
            value = np.hstack((value, variances))
        return value

    def get_impurity(self, node):
        if False:
            print('Hello World!')
        return self._tree.impurity[node]

    @property
    def max_depth(self):
        if False:
            for i in range(10):
                print('nop')
        return self._tree.max_depth

    @property
    def num_nodes(self):
        if False:
            print('Hello World!')
        return self._tree.node_count

    @property
    def root(self):
        if False:
            print('Hello World!')
        return 0

    @memoize_method(maxsize=1024)
    def rules(self, node):
        if False:
            print('Hello World!')
        if node != self.root:
            parent = self.parent(node)
            pr = OrderedDict([(r.attr_name, r) for r in self.rules(parent)])
            parent_attr = self.attribute(parent)
            parent_attr_cv = parent_attr.compute_value
            is_left_child = self.__left_child(parent) == node
            if isinstance(parent_attr_cv, Indicator) and hasattr(parent_attr_cv.variable, 'values'):
                values = parent_attr_cv.variable.values
                attr_name = parent_attr_cv.variable.name
                eq = not is_left_child * (len(values) != 2)
                value = values[abs(parent_attr_cv.value - is_left_child * (len(values) == 2))]
                new_rule = DiscreteRule(attr_name, eq, value)
                attr_name = attr_name + '_' + value
            else:
                attr_name = parent_attr.name
                sign = not is_left_child
                value = self._tree.threshold[self.parent(node)]
                new_rule = ContinuousRule(attr_name, sign, value, inclusive=is_left_child)
            if attr_name in pr:
                pr[attr_name] = pr[attr_name].merge_with(new_rule)
                pr.move_to_end(attr_name)
            else:
                pr[attr_name] = new_rule
            return list(pr.values())
        else:
            return []

    def short_rule(self, node):
        if False:
            i = 10
            return i + 15
        return self.rules(node)[0].description

    def attribute(self, node):
        if False:
            while True:
                i = 10
        feature_idx = self.splitting_attribute(node)
        if feature_idx == self.FEATURE_UNDEFINED:
            return None
        return self.domain.attributes[self.splitting_attribute(node)]

    def splitting_attribute(self, node):
        if False:
            return 10
        return self._tree.feature[node]

    @memoize_method(maxsize=1024)
    def leaves(self, node):
        if False:
            print('Hello World!')
        (start, stop) = self._subnode_range(node)
        if start == stop:
            return np.array([node], dtype=int)
        else:
            is_leaf = self._tree.children_left[start:stop] == self.NO_CHILD
            assert np.flatnonzero(is_leaf).size > 0
            return start + np.flatnonzero(is_leaf)

    def _subnode_range(self, node):
        if False:
            i = 10
            return i + 15
        '\n        Get the range of indices where there are subnodes of the given node.\n\n        See Also\n        --------\n        Orange.widgets.model.owclassificationtreegraph.OWTreeGraph\n        '

        def find_largest_idx(n):
            if False:
                for i in range(10):
                    print('nop')
            'It is necessary to locate the node with the largest index in the\n            children in order to get a good range. This is necessary with trees\n            that are not right aligned, which can happen when visualising\n            random forest trees.'
            if self._tree.children_left[n] == self.NO_CHILD:
                return n
            l_node = find_largest_idx(self._tree.children_left[n])
            r_node = find_largest_idx(self._tree.children_right[n])
            return max(l_node, r_node)
        right = left = node
        if self._tree.children_left[left] == self.NO_CHILD:
            assert self._tree.children_right[node] == self.NO_CHILD
            return (node, node)
        else:
            left = self._tree.children_left[left]
            right = find_largest_idx(right)
            return (left, right + 1)

    def get_samples_in_leaves(self):
        if False:
            i = 10
            return i + 15
        'Get an array of instance indices that belong to each leaf.\n\n        For a given dataset X, separate the instances out into an array, so\n        they are grouped together based on what leaf they belong to.\n\n        Examples\n        --------\n        Given a tree with two leaf nodes ( A <- R -> B ) and the dataset X =\n        [ 10, 20, 30, 40, 50, 60 ], where 10, 20 and 40 belong to leaf A, and\n        the rest to leaf B, the following structure will be returned (where\n        array is the numpy array):\n        [array([ 0, 1, 3 ]), array([ 2, 4, 5 ])]\n\n        The first array represents the indices of the values that belong to the\n        first leaft, so calling X[ 0, 1, 3 ] = [ 10, 20, 40 ]\n\n        Parameters\n        ----------\n        data\n            A matrix containing the data instances.\n\n        Returns\n        -------\n        np.array\n            The indices of instances belonging to a given leaf.\n\n        '

        def assign(node_id, indices):
            if False:
                return 10
            if self._tree.children_left[node_id] == self.NO_CHILD:
                return [indices]
            else:
                feature_idx = self._tree.feature[node_id]
                thresh = self._tree.threshold[node_id]
                column = self.instances_transformed.X[indices, feature_idx]
                leftmask = column <= thresh
                leftind = assign(self._tree.children_left[node_id], indices[leftmask])
                rightind = assign(self._tree.children_right[node_id], indices[~leftmask])
                return list.__iadd__(leftind, rightind)
        if self._all_leaves is not None:
            return self._all_leaves
        (n, _) = self.instances.X.shape
        items = np.arange(n, dtype=int)
        leaf_indices = assign(0, items)
        self._all_leaves = leaf_indices
        return leaf_indices

    def get_instances_in_nodes(self, nodes):
        if False:
            while True:
                i = 10
        if not isinstance(nodes, (list, tuple)):
            nodes = [nodes]
        node_leaves = [self.leaves(n) for n in nodes]
        if len(node_leaves) > 0:
            node_leaves = np.unique(np.hstack(node_leaves))
            all_leaves = self.leaves(self.root)
            indices = np.searchsorted(all_leaves, node_leaves)
            leaf_samples = self.get_samples_in_leaves()
            leaf_samples = [leaf_samples[i] for i in indices]
            indices = np.hstack(leaf_samples)
        else:
            indices = []
        return self.instances[indices] if len(indices) else None

    def get_indices(self, nodes):
        if False:
            return 10
        if not isinstance(nodes, (list, tuple)):
            nodes = [nodes]
        node_leaves = [self.leaves(n) for n in nodes]
        if len(node_leaves) > 0:
            node_leaves = np.unique(np.hstack(node_leaves))
            all_leaves = self.leaves(self.root)
            return np.searchsorted(all_leaves, node_leaves)
        return []