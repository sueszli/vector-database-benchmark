from collections import OrderedDict
from .Node import Node

def isNodeClass(cls):
    if False:
        while True:
            i = 10
    try:
        if not issubclass(cls, Node):
            return False
    except:
        return False
    return hasattr(cls, 'nodeName')

class NodeLibrary:
    """
    A library of flowchart Node types. Custom libraries may be built to provide 
    each flowchart with a specific set of allowed Node types.
    """

    def __init__(self):
        if False:
            print('Hello World!')
        self.nodeList = OrderedDict()
        self.nodeTree = OrderedDict()

    def addNodeType(self, nodeClass, paths, override=False):
        if False:
            while True:
                i = 10
        "\n        Register a new node type. If the type's name is already in use,\n        an exception will be raised (unless override=True).\n        \n        ============== =========================================================\n        **Arguments:**\n        \n        nodeClass      a subclass of Node (must have typ.nodeName)\n        paths          list of tuples specifying the location(s) this \n                       type will appear in the library tree.\n        override       if True, overwrite any class having the same name\n        ============== =========================================================\n        "
        if not isNodeClass(nodeClass):
            raise Exception('Object %s is not a Node subclass' % str(nodeClass))
        name = nodeClass.nodeName
        if not override and name in self.nodeList:
            raise Exception("Node type name '%s' is already registered." % name)
        self.nodeList[name] = nodeClass
        for path in paths:
            root = self.nodeTree
            for n in path:
                if n not in root:
                    root[n] = OrderedDict()
                root = root[n]
            root[name] = nodeClass

    def getNodeType(self, name):
        if False:
            i = 10
            return i + 15
        try:
            return self.nodeList[name]
        except KeyError:
            raise Exception("No node type called '%s'" % name)

    def getNodeTree(self):
        if False:
            return 10
        return self.nodeTree

    def copy(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return a copy of this library.\n        '
        lib = NodeLibrary()
        lib.nodeList = self.nodeList.copy()
        lib.nodeTree = self.treeCopy(self.nodeTree)
        return lib

    @staticmethod
    def treeCopy(tree):
        if False:
            print('Hello World!')
        copy = OrderedDict()
        for (k, v) in tree.items():
            if isNodeClass(v):
                copy[k] = v
            else:
                copy[k] = NodeLibrary.treeCopy(v)
        return copy

    def reload(self):
        if False:
            print('Hello World!')
        '\n        Reload Node classes in this library.\n        '
        raise NotImplementedError()