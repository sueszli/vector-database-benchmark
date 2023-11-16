"""Classes corresponding to CDAO trees.

See classes in ``Bio.Nexus``: Trees.Tree, Trees.NodeData, and Nodes.Chain.
"""
from Bio.Phylo import BaseTree

class Tree(BaseTree.Tree):
    """CDAO Tree object."""

    def __init__(self, root=None, rooted=False, id=None, name=None, weight=1.0):
        if False:
            while True:
                i = 10
        'Initialize value of for the CDAO tree object.'
        BaseTree.Tree.__init__(self, root=root or Clade(), rooted=rooted, id=id, name=name)
        self.weight = weight
        self.attributes = []

class Clade(BaseTree.Clade):
    """CDAO Clade (sub-tree) object."""

    def __init__(self, branch_length=1.0, name=None, clades=None, confidence=None, comment=None):
        if False:
            while True:
                i = 10
        'Initialize values for the CDAO Clade object.'
        BaseTree.Clade.__init__(self, branch_length=branch_length, name=name, clades=clades, confidence=confidence)
        self.comment = comment
        self.attributes = []
        self.tu_attributes = []
        self.edge_attributes = []