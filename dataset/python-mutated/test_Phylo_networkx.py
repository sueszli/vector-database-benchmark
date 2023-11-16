"""Unit tests for Bio.Phylo functions with external dependencies."""
import unittest
from Bio import Phylo
from Bio import MissingExternalDependencyError
try:
    import networkx
except ImportError:
    raise MissingExternalDependencyError('Install networkx if you wish to use it with Bio.Phylo') from None
EX_DOLLO = 'PhyloXML/o_tol_332_d_dollo.xml'
EX_APAF = 'PhyloXML/apaf.xml'

class UtilTests(unittest.TestCase):
    """Tests for various utility functions."""

    def test_to_networkx(self):
        if False:
            print('Hello World!')
        'Tree to Graph conversion, if networkx is available.'
        tree = Phylo.read(EX_DOLLO, 'phyloxml')
        G = Phylo.to_networkx(tree)
        self.assertEqual(len(G.nodes()), 659)
if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    unittest.main(testRunner=runner)