from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
from turicreate.util import _raise_error_if_not_of_type
from turicreate.toolkits._internal_utils import _numeric_param_check_range
import sys as _sys
if _sys.version_info.major == 3:
    long = int

class Node(object):

    def __init__(self, node_id, split_feature, value, node_type, left_id=None, right_id=None, missing_id=None):
        if False:
            return 10
        '\n        Simple class to make a node for a tree.\n\n        Parameters\n        ----------\n        node_id : A unique id for the node. Do not build a tree with 2 nodes\n                  that have the same id.\n\n        split_feature: The feature on which this node does the splitting.\n\n        value:  The prediction value (if leaf node), or the split value if\n                an intermediate node.\n\n        node_type: The node type; float, str, int, or leaf.\n        '
        self.node_id = node_id
        if split_feature is not None:
            self.split_feature_column = split_feature[0]
            self.split_feature_index = split_feature[1]
        else:
            self.split_feature_column = None
            self.split_feature_index = None
        self.node_type = node_type
        is_leaf = node_type == u'leaf'
        self.is_leaf = is_leaf
        self.value = value
        self.left = None
        self.missing = None
        self.right = None
        self.parent = None
        self.left_id = left_id
        self.right_id = right_id
        self.missing_id = missing_id
        self.parent_id = None

    def __repr__(self):
        if False:
            return 10
        out = ''
        out += '\nNode Id: %s' % self.node_id
        out += '\n'
        out += '\nSplit feature (column) : %s' % self.split_feature_column
        out += '\nSplit feature (index)  : %s' % self.split_feature_index
        out += '\nLeft                   : %s' % self.left_id
        out += '\nRight                  : %s' % self.right_id
        out += '\nMissing                : %s' % self.missing_id
        out += '\nLeaf?                  : %s' % self.is_leaf
        out += '\nValue                  : %s' % self.value
        return out

    def get_decision(self, child, is_missing=False):
        if False:
            while True:
                i = 10
        '\n        Get the decision from this node to a child node.\n\n        Parameters\n        ----------\n        child: Node\n            A child node of this node.\n\n        Returns\n        -------\n        dict: A dictionary that describes how to get from this node to the\n        child node.\n        '
        value = self.value
        feature = self.split_feature_column
        index = self.split_feature_index
        if not is_missing:
            if self.left_id == child.node_id:
                if self.node_type in ['float', 'integer']:
                    sign = '<'
                else:
                    sign = '='
            elif self.node_type in ['float', 'integer']:
                sign = '>='
            else:
                sign = '!='
        else:
            sign = 'missing'
            value = None
        return {'node_id': self.node_id, 'node_type': self.node_type, 'feature': feature, 'index': index, 'sign': sign, 'value': value, 'child_id': child.node_id, 'is_missing': is_missing}

    def to_dict(self):
        if False:
            print('Hello World!')
        '\n        Return the node as a dictionary.\n\n        Returns\n        -------\n        dict: All the attributes of this node as a dictionary (minus the left\n              and right).\n        '
        out = {}
        for key in self.__dict__.keys():
            if key not in ['left', 'right', 'missing', 'parent']:
                out[key] = self.__dict__[key]
        return out

    def __eq__(self, node):
        if False:
            while True:
                i = 10
        return self.node_id == node.node_id and self.value == node.value and (self.split_feature_column == node.split_feature_column) and (self.is_leaf == node.is_leaf) and (self.left_id == node.left_id) and (self.missing_id == node.missing_id) and (self.right_id == node.right_id) and (self.num_examples == node.num_examples)

class DecisionTree:

    def __init__(self):
        if False:
            print('Hello World!')
        "\n        A simple pure python wrapper around a GLC decision tree object.\n\n        The tree can be obtained directly from trees_json parameter in any of\n        GLC model objects (boosted trees, random forests, and decision trees).\n\n        Parameters\n        ----------\n        model : Tree/Tree ensemble model. Can be any tree of type boosted\n                trees, random forests, or decision tree.\n\n        tree_id : Tree id in the ensemble to export.\n\n        Examples\n        --------\n        .. sourcecode:: python\n\n            >>> url = 'https://static.turi.com/datasets/xgboost/mushroom.csv'\n            >>> data = tc.SFrame.read_csv(url)\n\n            >>> train, test = data.random_split(0.8)\n            >>> model = tc.boosted_trees_classifier.create(train,\n            ...                  target='label', validation_set = None)\n\n            >>> tree = model._get_tree()\n        "
        pass

    @classmethod
    def from_model(cls, model, tree_id=0):
        if False:
            print('Hello World!')
        import turicreate as _tc
        import json as _json
        _raise_error_if_not_of_type(tree_id, [int, long], 'tree_id')
        _numeric_param_check_range('tree_id', tree_id, 0, model.num_trees - 1)
        tree = DecisionTree()
        tree_str = _tc.extensions._xgboost_get_tree(model.__proxy__, tree_id)
        metadata_mapping = _tc.extensions._supervised_learning._get_metadata_mapping(model.__proxy__)
        trees_json = _json.loads(tree_str)
        tree._make_tree(trees_json, metadata_mapping)
        tree.root_id = 0
        for key in {'num_examples', 'num_features', 'num_unpacked_features', 'max_depth'}:
            setattr(tree, key, model._get(key))
        return tree

    def __repr__(self):
        if False:
            while True:
                i = 10
        if hasattr(self, 'node'):
            return 'Uninitialized decision tree.'
        out = ''
        out += 'Python decision tree'
        out += '\n'
        out += '\nNumber of examples          : %s' % self.num_examples
        out += '\nNumber of feature columns   : %s' % self.num_features
        out += '\nNumber of unpacked features : %s' % self.num_unpacked_features
        out += '\n'
        out += '\nMaximum tree depth          : %s' % self.max_depth
        out += '\nNumber of nodes             : %s' % self.num_nodes
        return out

    def _parse_tree_json_vertices(self, vertices, metadata_mapping):
        if False:
            i = 10
            return i + 15
        nodes = {}
        for v in vertices:
            node_id = v.get('id', None)
            split_feature = v.get('name', None)
            if split_feature is not None:
                idx = int(split_feature.strip('{').strip('}'))
                split_feature = metadata_mapping[idx]
            value = v.get('value', None)
            node_type = v.get('type', None)
            left_id = v.get('yes_child', None)
            right_id = v.get('no_child', None)
            missing_id = v.get('missing_child', None)
            nodes[node_id] = Node(node_id, split_feature, value, node_type, left_id, right_id, missing_id)
        return nodes

    def _make_tree(self, trees_json, metadata_mapping):
        if False:
            for i in range(10):
                print('nop')
        vertices = trees_json['vertices']
        self.nodes = self._parse_tree_json_vertices(vertices, metadata_mapping)
        edges = []
        for (nid, node) in self.nodes.items():
            if not node.is_leaf:
                e = [{'src': node.node_id, 'dst': node.left_id, 'value': 'left'}, {'src': node.node_id, 'dst': node.right_id, 'value': 'right'}, {'src': node.node_id, 'dst': node.missing_id, 'value': 'missing'}]
                edges += e
        for e in edges:
            src = e['src']
            dst = e['dst']
            value = e['value']
            if value == 'left':
                self.nodes[src].left_id = dst
                self.nodes[src].left = self.nodes[dst]
            elif value == 'right':
                self.nodes[src].right_id = dst
                self.nodes[src].right = self.nodes[dst]
            else:
                self.nodes[src].missing_id = dst
                self.nodes[src].missing = self.nodes[dst]
            self.nodes[dst].parent_id = src
            self.nodes[dst].parent = self.nodes[src]
        self.num_nodes = len(self.nodes)
        for (nid, n) in self.nodes.items():
            if n.parent is None:
                self._root_id = n.node_id
                break

    def to_json(self, root_id=0, output={}):
        if False:
            print('Hello World!')
        "\n        Recursive function to dump this tree as a json blob.\n\n        Parameters\n        ----------\n        root_id: Root id of the sub-tree\n        output: Carry over output from the previous sub-trees.\n\n        Returns\n        -------\n        dict: A tree in JSON format. Starts at the root node and recursively\n        represents each node in JSON.\n\n        - node_id              : ID of the node.\n        - left_id              : ID of left child (None if it doesn't exist).\n        - right_id             : ID of right child (None if it doesn't exist).\n        - split_feature_column : Feature column on which a decision is made.\n        - split_feature_index  : Feature index (within that column) on which the\n                                 decision is made.\n        - is_leaf              : Is this node a leaf node?\n        - node_type            : Node type (categorical, numerical, leaf etc.)\n        - value                : Prediction (if leaf), decision split point\n                                 (if not leaf).\n        - left                 : JSON representation of the left node.\n        - right                : JSON representation of the right node.\n\n        Examples\n        --------\n        .. sourcecode:: python\n\n            >>> tree.to_json()  # Leaf node\n            {'is_leaf': False,\n             'left': {'is_leaf': True,\n                      'left_id': None,\n                      'node_id': 115,\n                      'node_type': u'leaf',\n                      'parent_id': 60,\n                      'right_id': None,\n                      'split_feature_column': None,\n                      'split_feature_index': None,\n                      'value': 0.436364},\n             'left_id': 115,\n             'node_id': 60,\n             'node_type': u'float',\n             'parent_id': 29,\n             'right': {'is_leaf': True,\n                       'left_id': None,\n                       'node_id': 116,\n                       'node_type': u'leaf',\n                       'parent_id': 60,\n                       'right_id': None,\n                       'split_feature_column': None,\n                       'split_feature_index': None,\n                       'value': -0.105882},\n             'right_id': 116,\n             'split_feature_column': 'Quantity_features_14',\n             'split_feature_index': 'count_sum',\n             'value': 22.5}\n        "
        _raise_error_if_not_of_type(root_id, [int, long], 'root_id')
        _numeric_param_check_range('root_id', root_id, 0, self.num_nodes - 1)
        node = self.nodes[root_id]
        output = node.to_dict()
        if node.left_id is not None:
            j = node.left_id
            output['left'] = self.to_json(j, output)
        if node.right_id is not None:
            j = node.right_id
            output['right'] = self.to_json(j, output)
        return output

    def get_prediction_score(self, node_id):
        if False:
            i = 10
            return i + 15
        '\n        Return the prediction score (if leaf node) or None if its an\n        intermediate node.\n\n        Parameters\n        ----------\n        node_id: id of the node to get the prediction value.\n\n        Returns\n        -------\n        float or None: returns float value of prediction if leaf node and None\n        if not.\n\n        Examples\n        --------\n        .. sourcecode:: python\n\n            >>> tree.get_prediction_score(120)  # Leaf node\n            0.251092\n\n            >>> tree.get_prediction_score(120)  # Not a leaf node\n            None\n\n        '
        _raise_error_if_not_of_type(node_id, [int, long], 'node_id')
        _numeric_param_check_range('node_id', node_id, 0, self.num_nodes - 1)
        node = self.nodes[node_id]
        return None if node.is_leaf is False else node.value

    def get_prediction_path(self, node_id, missing_id=[]):
        if False:
            i = 10
            return i + 15
        "\n        Return the prediction path from this node to the parent node.\n\n        Parameters\n        ----------\n        node_id    : id of the node to get the prediction path.\n        missing_id : Additional info that contains nodes with missing features.\n\n        Returns\n        -------\n        list: The list of decisions (top to bottom) from the root to this node.\n\n        Examples\n        --------\n        .. sourcecode:: python\n\n            >>> tree.get_prediction_score(5)  # Any node\n             [{'child_id': 2,\n               'feature': 'Quantity_features_90',\n               'index': 'sum_timegaplast_gap',\n               'node_id': 0,\n               'sign': '>',\n               'value': 53.5},\n              {'child_id': 5,\n               'feature': 'Quantity_features_90',\n               'index': 'sum_sum',\n               'node_id': 2,\n               'sign': '<=',\n               'value': 146.5}]\n        "
        _raise_error_if_not_of_type(node_id, [int, long], 'node_id')
        _numeric_param_check_range('node_id', node_id, 0, self.num_nodes - 1)

        def _deduplicate_path(path):
            if False:
                return 10
            s_nodes = {}
            s_path = []
            for node in path:
                feature = node['feature']
                index = node['index']
                if (feature, index) not in s_nodes:
                    s_nodes[feature, index] = node
                    s_path.append(node)
                else:
                    s_node = s_nodes[feature, index]
                    s_sign = s_node['sign']
                    sign = node['sign']
                    value = node['value']
                    if s_sign == '<':
                        if sign == '>=':
                            s_node['value'] = [value, s_node['value']]
                            s_node['sign'] = 'in'
                        elif sign == '<':
                            s_node['value'] = value
                    elif s_sign == '>=':
                        if sign == '>=':
                            s_node['value'] = value
                        elif sign == '<':
                            s_node['value'] = [s_node['value'], value]
                            s_node['sign'] = 'in'
                    elif s_sign == 'in':
                        if sign == '>=':
                            s_node['value'][0] = value
                        elif sign == '<':
                            s_node['value'][1] = value
            return s_path
        path = []
        node = self.nodes[node_id]
        while node.parent is not None:
            parent = node.parent
            is_missing = node.node_id in missing_id
            path.insert(0, parent.get_decision(node, is_missing))
            node = node.parent
        return _deduplicate_path(path)

    @property
    def root(self):
        if False:
            while True:
                i = 10
        return None if self._root_id is None else self.nodes[self.root_id]

    def __getitem__(self, node_id):
        if False:
            return 10
        return self.nodes[node_id]

    def __iter__(self):
        if False:
            print('Hello World!')
        for x in self.nodes:
            yield self.nodes[x]