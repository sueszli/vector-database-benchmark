from dataclasses import dataclass, field
from typing import Dict, Optional

def X(val):
    if False:
        print('Hello World!')
    '\n    Compact way to write a leaf node\n    '
    return (val, [])

def XImportant(name):
    if False:
        for i in range(10):
            print('nop')
    'Compact way to write an important (run on PRs) leaf node'
    return (name, [('important', [X(True)])])

@dataclass
class Ver:
    """
    Represents a product with a version number
    """
    name: str
    version: str = ''

    def __str__(self):
        if False:
            while True:
                i = 10
        return self.name + self.version

@dataclass
class ConfigNode:
    parent: Optional['ConfigNode']
    node_name: str
    props: Dict[str, str] = field(default_factory=dict)

    def get_label(self):
        if False:
            return 10
        return self.node_name

    def get_children(self):
        if False:
            i = 10
            return i + 15
        return []

    def get_parents(self):
        if False:
            for i in range(10):
                print('nop')
        return self.parent.get_parents() + [self.parent.get_label()] if self.parent else []

    def get_depth(self):
        if False:
            return 10
        return len(self.get_parents())

    def get_node_key(self):
        if False:
            print('Hello World!')
        return '%'.join(self.get_parents() + [self.get_label()])

    def find_prop(self, propname, searched=None):
        if False:
            while True:
                i = 10
        '\n        Checks if its own dictionary has\n        the property, otherwise asks parent node.\n        '
        if searched is None:
            searched = []
        searched.append(self.node_name)
        if propname in self.props:
            return self.props[propname]
        elif self.parent:
            return self.parent.find_prop(propname, searched)
        else:
            return None

def dfs_recurse(node, leaf_callback=lambda x: None, discovery_callback=lambda x, y, z: None, child_callback=lambda x, y: None, sibling_index=0, sibling_count=1):
    if False:
        for i in range(10):
            print('nop')
    discovery_callback(node, sibling_index, sibling_count)
    node_children = node.get_children()
    if node_children:
        for (i, child) in enumerate(node_children):
            child_callback(node, child)
            dfs_recurse(child, leaf_callback, discovery_callback, child_callback, i, len(node_children))
    else:
        leaf_callback(node)

def dfs(toplevel_config_node):
    if False:
        print('Hello World!')
    config_list = []

    def leaf_callback(node):
        if False:
            return 10
        config_list.append(node)
    dfs_recurse(toplevel_config_node, leaf_callback)
    return config_list