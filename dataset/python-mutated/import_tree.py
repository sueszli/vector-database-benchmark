"""
Tree structure for resolving imports.
"""
from __future__ import annotations
from enum import Enum
import typing
if typing.TYPE_CHECKING:
    from openage.convert.entity_object.export.formats.nyan_file import NyanFile
    from openage.nyan.nyan_structs import NyanObject

class NodeType(Enum):
    """
    Types for nodes.
    """
    ROOT = 'r'
    FILESYS = 'f'
    OBJECT = 'o'
    NESTED = 'no'

class Node:
    """
    Node in the import tree. This can be a directory, a file
    or an object.
    """
    __slots__ = ('name', 'node_type', 'parent', 'depth', 'children', 'alias')

    def __init__(self, name: str, node_type: NodeType, parent):
        if False:
            print('Hello World!')
        '\n        Create a node for an import tree.\n\n        :param name: Name of the node.\n        :type name: str\n        :param node_type: Type of the node.\n        :type node_type: NodeType\n        :param parent: Parent node of this node.\n        :type parent: Node\n        '
        self.name = name
        self.node_type = node_type
        self.parent: Node = parent
        if not self.parent and self.node_type is not NodeType.ROOT:
            raise TypeError('Only node with type ROOT are allowed to have no parent')
        self.depth = 0
        if self.node_type is NodeType.ROOT:
            self.depth = 0
        else:
            self.depth = self.parent.depth + 1
        self.children = {}
        self.alias = ''

    def add_child(self, child_node) -> None:
        if False:
            print('Hello World!')
        '\n        Adds a child node to this node.\n        '
        self.children.update({child_node.name: child_node})

    def has_ancestor(self, ancestor_node, max_distance: int=128) -> bool:
        if False:
            while True:
                i = 10
        '\n        Checks is the node has a given node as ancestor.\n\n        :param ancestor_node: Ancestor candidate node.\n        :type ancestor_node: Node\n        '
        current_node = self
        distance = 0
        while distance < max_distance:
            if current_node.parent is ancestor_node:
                return True
            if not current_node.parent:
                return False
            current_node = current_node.parent
            distance += 1
        return False

    def has_child(self, name: str) -> bool:
        if False:
            while True:
                i = 10
        '\n        Checks if a child with the given name exists.\n\n        :param name: Name of the child node.\n        :type name: str\n        '
        return name in self.children

    def get_child(self, name: str):
        if False:
            print('Hello World!')
        '\n        Returns the child noe with the given name.\n\n        :param name: Name of the child node.\n        :type name: str\n        '
        return self.children[name]

    def get_fqon(self) -> tuple[str]:
        if False:
            while True:
                i = 10
        '\n        Get the fqon that is associated with this node by traversing the tree upwards.\n        '
        current_node = self
        fqon = []
        while current_node.node_type is not NodeType.ROOT:
            fqon.insert(0, current_node.name)
            current_node = current_node.parent
        return tuple(fqon)

    def set_alias(self, alias: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Give this node an alias name.\n\n        :param alias: Alias for the node.\n        :type alias: str\n        '
        if self.node_type is not NodeType.FILESYS:
            raise TypeError('Only nodes of type FILESYS can have aliases')
        self.alias = alias

class ImportTree:
    """
    Tree for storing nyan object references.
    """
    __slots__ = ('root', 'alias_nodes', 'import_nodes')

    def __init__(self):
        if False:
            while True:
                i = 10
        self.root = Node('', NodeType.ROOT, None)
        self.alias_nodes: set[Node] = set()
        self.import_nodes: set[Node] = set()

    def add_alias(self, fqon: tuple[str], alias: str) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Adds an alias to the node with the specified fqon.\n\n        :param fqon: Identifier of the node.\n        :type fqon: tuple[str]\n        :param alias: Alias for the node.\n        :type alias: str\n        '
        current_node = self.root
        for node_str in fqon:
            try:
                current_node = current_node.get_child(node_str)
            except KeyError:
                return
        current_node.set_alias(alias)

    def clear_marks(self) -> None:
        if False:
            print('Hello World!')
        '\n        Remove all alias marks from the tree.\n        '
        self.alias_nodes.clear()
        self.import_nodes.clear()

    def expand_from_file(self, nyan_file: NyanFile) -> None:
        if False:
            return 10
        '\n        Expands the tree from a nyan file.\n\n        :param nyan_file: File with nyan objects.\n        :type nyan_file: NyanFile\n        '
        current_node = self.root
        fqon = nyan_file.get_fqon()
        node_type = NodeType.FILESYS
        for node_str in fqon:
            if current_node.has_child(node_str):
                current_node = current_node.get_child(node_str)
            else:
                new_node = Node(node_str, node_type, current_node)
                current_node.add_child(new_node)
                current_node = new_node
        for nyan_object in nyan_file.nyan_objects:
            self.expand_from_object(nyan_object)

    def expand_from_object(self, nyan_object: NyanObject) -> None:
        if False:
            print('Hello World!')
        '\n        Expands the tree from a nyan object.\n\n        :param nyan_object: A nyan object.\n        :type nyan_object: NyanObject\n        '
        fqon = nyan_object.get_fqon()
        if fqon[0] != 'engine':
            current_node = self.root
            node_type = NodeType.OBJECT
            for node_str in fqon:
                if current_node.has_child(node_str):
                    current_node = current_node.get_child(node_str)
                else:
                    new_node = Node(node_str, node_type, current_node)
                    current_node.add_child(new_node)
                    current_node = new_node
        else:
            current_node = self.root
            index = 0
            while index < len(fqon):
                node_str = fqon[index]
                if current_node.has_child(node_str):
                    current_node = current_node.get_child(node_str)
                else:
                    if node_str[0].islower():
                        node_type = NodeType.FILESYS
                    else:
                        node_type = NodeType.OBJECT
                    new_node = Node(node_str, node_type, current_node)
                    current_node.add_child(new_node)
                    current_node = new_node
                index += 1
        self._expand_nested_objects(nyan_object)

    def _expand_nested_objects(self, nyan_object: NyanObject):
        if False:
            while True:
                i = 10
        '\n        Recursively search the nyan objects for nested objects\n        '
        unsearched_objects = []
        unsearched_objects.extend(nyan_object.get_nested_objects())
        found_nested_objects = []
        while len(unsearched_objects) > 0:
            current_nested_object = unsearched_objects[0]
            unsearched_objects.extend(current_nested_object.get_nested_objects())
            found_nested_objects.append(current_nested_object)
            unsearched_objects.remove(current_nested_object)
        for nested_object in found_nested_objects:
            current_node = self.root
            node_type = NodeType.NESTED
            fqon = nested_object.get_fqon()
            for node_str in fqon:
                if current_node.has_child(node_str):
                    current_node = current_node.get_child(node_str)
                else:
                    new_node = Node(node_str, node_type, current_node)
                    current_node.add_child(new_node)
                    current_node = new_node

    def get_alias_dict(self) -> dict[str, tuple[str]]:
        if False:
            i = 10
            return i + 15
        '\n        Get the fqons of the nodes that are used for aliases, i.e. fqons of all\n        nodes in self.alias_nodes. The dict can be used for creating imports\n        of a nyan file.\n\n        Call this function after all object references in a file have been\n        searched for aliases with get_alias_fqon().\n        '
        aliases = {}
        for current_node in self.alias_nodes:
            if current_node.alias in aliases:
                raise ValueError(f'duplicate alias: {current_node.alias}')
            aliases.update({current_node.alias: current_node.get_fqon()})
        aliases = dict(sorted(aliases.items(), key=lambda item: item[1]))
        return aliases

    def get_import_list(self) -> list[tuple[str]]:
        if False:
            while True:
                i = 10
        '\n        Get the fqons of the nodes that are plain imports, i.e. fqons of all\n        nodes in self.import_nodes. The dict can be used for creating imports\n        of a nyan file.\n\n        Call this function after all object references in a file have been\n        searched for aliases with get_alias_fqon().\n        '
        imports = []
        for current_node in self.import_nodes:
            imports.append(current_node.get_fqon())
        imports.sort()
        return imports

    def get_alias_fqon(self, fqon: tuple[str], namespace: tuple[str]=None) -> tuple[str]:
        if False:
            return 10
        '\n        Find the (shortened) fqon by traversing the tree to the fqon node and\n        then going upwards until an alias is found.\n\n        :param fqon: Object reference for which an alias should be found.\n        :type fqon: tuple[str]\n        :param namespace: Identifier of a namespace. If this is a (nested) object,\n                          we check if the fqon is in the namespace before\n                          searching for an alias.\n        :type namespace: tuple[str]\n        '
        if namespace:
            current_node = self.root
            if len(namespace) <= len(fqon):
                for (index, namespace_part) in enumerate(namespace):
                    current_node = current_node.get_child(namespace_part)
                    if namespace_part != fqon[index]:
                        break
                else:
                    if current_node.node_type in (NodeType.OBJECT, NodeType.NESTED):
                        return (fqon[-1],)
        current_node = self.root
        for part in fqon:
            current_node = current_node.get_child(part)
        sfqon = []
        file_node = None
        while current_node.depth > 0:
            if file_node is None and current_node.node_type == NodeType.FILESYS:
                file_node = current_node
            if current_node.alias:
                sfqon.insert(0, current_node.alias)
                self.alias_nodes.add(current_node)
                break
            sfqon.insert(0, current_node.name)
            current_node = current_node.parent
        else:
            if file_node:
                self.import_nodes.add(file_node)
        return tuple(sfqon)