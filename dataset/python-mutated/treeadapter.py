"""Base tree adapter class with common methods needed for visualisations."""
from abc import ABCMeta, abstractmethod
from functools import reduce
from operator import add
import random

class BaseTreeAdapter(metaclass=ABCMeta):
    """Base class for tree representation.

    Any subclass should implement the methods listed in this base class. Note
    that some simple methods do not need to reimplemented e.g. is_leaf since
    it that is the opposite of has_children.

    """
    ROOT_PARENT = None
    NO_CHILD = -1
    FEATURE_UNDEFINED = -2

    def __init__(self, model):
        if False:
            print('Hello World!')
        self.model = model
        self.domain = model.domain
        self.instances = model.instances
        self.instances_transformed = self.instances.transform(self.domain)

    @abstractmethod
    def weight(self, node):
        if False:
            print('Hello World!')
        'Get the weight of the given node.\n\n        The weights of the children always sum up to 1.\n\n        Parameters\n        ----------\n        node : object\n            The label of the node.\n\n        Returns\n        -------\n        float\n            The weight of the node relative to its siblings.\n\n        '

    @abstractmethod
    def num_samples(self, node):
        if False:
            return 10
        'Get the number of samples that a given node contains.\n\n        Parameters\n        ----------\n        node : object\n            A unique identifier of a node.\n\n        Returns\n        -------\n        int\n\n        '

    @abstractmethod
    def parent(self, node):
        if False:
            return 10
        'Get the parent of a given node or ROOT_PARENT if the node is the root.\n\n        Parameters\n        ----------\n        node : object\n\n        Returns\n        -------\n        object\n\n        '

    @abstractmethod
    def has_children(self, node):
        if False:
            return 10
        'Check if the given node has any children.\n\n        Parameters\n        ----------\n        node : object\n\n        Returns\n        -------\n        bool\n\n        '

    def is_leaf(self, node):
        if False:
            print('Hello World!')
        'Check if the given node is a leaf node.\n\n        Parameters\n        ----------\n        node : object\n\n        Returns\n        -------\n        object\n\n        '
        return not self.has_children(node)

    @abstractmethod
    def children(self, node):
        if False:
            for i in range(10):
                print('nop')
        'Get all the children of a given node.\n\n        Parameters\n        ----------\n        node : object\n\n        Returns\n        -------\n        Iterable[object]\n            A iterable object containing the labels of the child nodes.\n\n        '

    def reverse_children(self, node):
        if False:
            while True:
                i = 10
        'Reverse children of a given node.\n\n        Parameters\n        ----------\n        node : object\n        '

    def shuffle_children(self):
        if False:
            i = 10
            return i + 15
        "Randomly shuffle node's children in the entire tree.\n        "

    @abstractmethod
    def get_distribution(self, node):
        if False:
            while True:
                i = 10
        'Get the distribution of types for a given node.\n\n        This may be the number of nodes that belong to each different classe in\n        a node.\n\n        Parameters\n        ----------\n        node : object\n\n        Returns\n        -------\n        Iterable[int, ...]\n            The return type is an iterable with as many fields as there are\n            different classes in the given node. The values of the fields are\n            the number of nodes that belong to a given class inside the node.\n\n        '

    @abstractmethod
    def get_impurity(self, node):
        if False:
            print('Hello World!')
        'Get the impurity of a given node.\n\n        Parameters\n        ----------\n        node : object\n\n        Returns\n        -------\n        object\n\n        '

    @abstractmethod
    def rules(self, node):
        if False:
            return 10
        'Get a list of rules that define the given node.\n\n        Parameters\n        ----------\n        node : object\n\n        Returns\n        -------\n        Iterable[Rule]\n            A list of Rule objects, can be of any type.\n\n        '

    @abstractmethod
    def short_rule(self, node):
        if False:
            i = 10
            return i + 15
        pass

    @abstractmethod
    def attribute(self, node):
        if False:
            return 10
        'Get the attribute that splits the given tree.\n\n        Parameters\n        ----------\n        node\n\n        Returns\n        -------\n\n        '

    def is_root(self, node):
        if False:
            while True:
                i = 10
        'Check if a given node is the root node.\n\n        Parameters\n        ----------\n        node\n\n        Returns\n        -------\n\n        '
        return node == self.root

    @abstractmethod
    def leaves(self, node):
        if False:
            print('Hello World!')
        'Get all the leavse that belong to the subtree of a given node.\n\n        Parameters\n        ----------\n        node\n\n        Returns\n        -------\n\n        '

    @abstractmethod
    def get_instances_in_nodes(self, dataset, nodes):
        if False:
            print('Hello World!')
        'Get all the instances belonging to a set of nodes for a given\n        dataset.\n\n        Parameters\n        ----------\n        dataset : Table\n            A Orange Table dataset.\n        nodes : iterable[node]\n            A list of tree nodes for which we want the instances.\n\n        Returns\n        -------\n\n        '

    @abstractmethod
    def get_indices(self, nodes):
        if False:
            print('Hello World!')
        pass

    @property
    @abstractmethod
    def max_depth(self):
        if False:
            for i in range(10):
                print('nop')
        'Get the maximum depth that the tree reaches.\n\n        Returns\n        -------\n        int\n\n        '

    @property
    @abstractmethod
    def num_nodes(self):
        if False:
            for i in range(10):
                print('nop')
        'Get the total number of nodes that the tree contains.\n\n        This does not mean the number of samples inside the entire tree, just\n        the number of nodes.\n\n        Returns\n        -------\n        int\n\n        '

    @property
    @abstractmethod
    def root(self):
        if False:
            i = 10
            return i + 15
        'Get the label of the root node.\n\n        Returns\n        -------\n        object\n\n        '

class TreeAdapter(BaseTreeAdapter):

    def weight(self, node):
        if False:
            while True:
                i = 10
        return len(node.subset) / len(node.parent.subset)

    def num_samples(self, node):
        if False:
            i = 10
            return i + 15
        return len(node.subset)

    def parent(self, node):
        if False:
            print('Hello World!')
        return node.parent

    def has_children(self, node):
        if False:
            return 10
        return any(node.children)

    def is_leaf(self, node):
        if False:
            return 10
        return not any(node.children)

    def children(self, node):
        if False:
            while True:
                i = 10
        return [child for child in node.children if child is not None]

    def reverse_children(self, node):
        if False:
            for i in range(10):
                print('nop')
        node.children = node.children[::-1]

    def shuffle_children(self):
        if False:
            return 10

        def _shuffle_children(node):
            if False:
                while True:
                    i = 10
            if node and node.children:
                random.shuffle(node.children)
                for c in node.children:
                    _shuffle_children(c)
        _shuffle_children(self.root)

    def get_distribution(self, node):
        if False:
            print('Hello World!')
        return [node.value]

    def get_impurity(self, node):
        if False:
            print('Hello World!')
        raise NotImplementedError

    def rules(self, node):
        if False:
            while True:
                i = 10
        return self.model.rule(node)

    def short_rule(self, node):
        if False:
            print('Hello World!')
        return node.description

    def attribute(self, node):
        if False:
            print('Hello World!')
        return node.attr

    def leaves(self, node):
        if False:
            print('Hello World!')

        def _leaves(node):
            if False:
                i = 10
                return i + 15
            return reduce(add, map(_leaves, self.children(node)), []) or [node]
        return _leaves(node)

    def get_instances_in_nodes(self, nodes):
        if False:
            i = 10
            return i + 15
        from Orange import tree
        if isinstance(nodes, tree.Node):
            nodes = [nodes]
        return self.model.get_instances(nodes)

    def get_indices(self, nodes):
        if False:
            for i in range(10):
                print('nop')
        return self.model.get_indices(nodes)

    @property
    def max_depth(self):
        if False:
            while True:
                i = 10
        return self.model.depth()

    @property
    def num_nodes(self):
        if False:
            print('Hello World!')
        return self.model.node_count()

    @property
    def root(self):
        if False:
            return 10
        return self.model.root