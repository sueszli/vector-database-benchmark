"""Unit tests for the Bio.Phylo module."""
import os
import unittest
import tempfile
from io import StringIO
from Bio import Phylo
from Bio.Phylo import PhyloXML
EX_NEWICK = 'Nexus/int_node_labels.nwk'
EX_NEWICK2 = 'Nexus/test.new'
EX_NEXUS = 'Nexus/test_Nexus_input.nex'
EX_NEXUS2 = 'Nexus/bats.nex'
EX_NEWICK_BOM = 'Nexus/ByteOrderMarkFile.nwk'
EX_APAF = 'PhyloXML/apaf.xml'
EX_BCL2 = 'PhyloXML/bcl_2.xml'
EX_DIST = 'PhyloXML/distribution.xml'
EX_PHYLO = 'PhyloXML/phyloxml_examples.xml'

class IOTests(unittest.TestCase):
    """Tests for parsing and writing the supported formats."""

    def test_newick_read_single1(self):
        if False:
            for i in range(10):
                print('nop')
        'Read first Newick file with one tree.'
        tree = Phylo.read(EX_NEWICK, 'newick')
        self.assertEqual(len(tree.get_terminals()), 28)

    def test_newick_read_single2(self):
        if False:
            while True:
                i = 10
        'Read second Newick file with one tree.'
        tree = Phylo.read(EX_NEWICK2, 'newick')
        self.assertEqual(len(tree.get_terminals()), 33)
        self.assertEqual(tree.find_any('Homo sapiens').comment, 'modern human')
        self.assertEqual(tree.find_any('Equus caballus').comment, "wild horse; also 'Equus ferus caballus'")
        self.assertEqual(tree.root.confidence, 80)
        tree = Phylo.read(EX_NEWICK2, 'newick', comments_are_confidence=True)
        self.assertEqual(tree.root.confidence, 100)

    def test_newick_read_single3(self):
        if False:
            while True:
                i = 10
        'Read Nexus file with one tree.'
        tree = Phylo.read(EX_NEXUS2, 'nexus')
        self.assertEqual(len(tree.get_terminals()), 658)

    def test_unicode_exception(self):
        if False:
            print('Hello World!')
        'Read a Newick file with a unicode byte order mark (BOM).'
        with open(EX_NEWICK_BOM, encoding='utf-8') as handle:
            tree = Phylo.read(handle, 'newick')
        self.assertEqual(len(tree.get_terminals()), 3)

    def test_newick_read_multiple(self):
        if False:
            i = 10
            return i + 15
        'Parse a Nexus file with multiple trees.'
        trees = list(Phylo.parse(EX_NEXUS, 'nexus'))
        self.assertEqual(len(trees), 3)
        for tree in trees:
            self.assertEqual(len(tree.get_terminals()), 9)

    def test_newick_write(self):
        if False:
            return 10
        'Parse a Nexus file with multiple trees.'
        mem_file = StringIO()
        tree = Phylo.read(StringIO('(A,B,(C,D)E)F;'), 'newick')
        Phylo.write(tree, mem_file, 'newick')
        mem_file.seek(0)
        tree2 = Phylo.read(mem_file, 'newick')
        self.assertEqual(tree2.count_terminals(), 4)
        internal_names = {c.name for c in tree2.get_nonterminals() if c is not None}
        self.assertEqual(internal_names, {'E', 'F'})

    def test_newick_read_scinot(self):
        if False:
            i = 10
            return i + 15
        'Parse Newick branch lengths in scientific notation.'
        tree = Phylo.read(StringIO('(foo:1e-1,bar:0.1)'), 'newick')
        clade_a = tree.clade[0]
        self.assertEqual(clade_a.name, 'foo')
        self.assertAlmostEqual(clade_a.branch_length, 0.1)

    def test_phylo_read_extra(self):
        if False:
            for i in range(10):
                print('nop')
        'Additional tests to check correct parsing.'
        tree = Phylo.read(StringIO('(A:1, B:-2, (C:3, D:4):-2)'), 'newick')
        self.assertEqual(tree.distance('A'), 1)
        self.assertEqual(tree.distance('B'), -2)
        self.assertEqual(tree.distance('C'), 1)
        self.assertEqual(tree.distance('D'), 2)
        tree = Phylo.read(StringIO('((A:1, B:-2):-5, (C:3, D:4):-2)'), 'newick')
        self.assertEqual(tree.distance('A'), -4)
        self.assertEqual(tree.distance('B'), -7)
        self.assertEqual(tree.distance('C'), 1)
        self.assertEqual(tree.distance('D'), 2)
        tree = Phylo.read(StringIO('((:1, B:-2):-5, (C:3, D:4):-2)'), 'newick')
        distances = {-4.0: 1, -7.0: 1, 1: 1, 2: 1}
        for x in tree.get_terminals():
            entry = int(tree.distance(x))
            distances[entry] -= distances[entry]
            self.assertEqual(distances[entry], 0)
        tree = Phylo.read(StringIO('((:\n1\n,\n B:-2):-5, (C:3, D:4):-2);'), 'newick')
        distances = {-4.0: 1, -7.0: 1, 1: 1, 2: 1}
        for x in tree.get_terminals():
            entry = int(tree.distance(x))
            distances[entry] -= distances[entry]
            self.assertEqual(distances[entry], 0)

    def test_format_branch_length(self):
        if False:
            i = 10
            return i + 15
        'Custom format string for Newick branch length serialization.'
        tree = Phylo.read(StringIO('A:0.1;'), 'newick')
        mem_file = StringIO()
        Phylo.write(tree, mem_file, 'newick', format_branch_length='%.0e')
        value = mem_file.getvalue().strip()
        self.assertTrue(value.startswith('A:'))
        self.assertTrue(value.endswith(';'))
        self.assertEqual(value[2:-1], '%.0e' % 0.1)

    def test_convert(self):
        if False:
            print('Hello World!')
        'Convert a tree between all supported formats.'
        mem_file_1 = StringIO()
        mem_file_2 = StringIO()
        mem_file_3 = StringIO()
        Phylo.convert(EX_NEWICK, 'newick', mem_file_1, 'nexus')
        mem_file_1.seek(0)
        Phylo.convert(mem_file_1, 'nexus', mem_file_2, 'phyloxml')
        mem_file_2.seek(0)
        Phylo.convert(mem_file_2, 'phyloxml', mem_file_3, 'newick')
        mem_file_3.seek(0)
        tree = Phylo.read(mem_file_3, 'newick')
        self.assertEqual(len(tree.get_terminals()), 28)

    def test_convert_phyloxml_binary(self):
        if False:
            return 10
        'Try writing phyloxml to a binary handle; fail on Py3.'
        trees = Phylo.parse('PhyloXML/phyloxml_examples.xml', 'phyloxml')
        with tempfile.NamedTemporaryFile(mode='wb') as out_handle:
            self.assertRaises(TypeError, Phylo.write, trees, out_handle, 'phyloxml')

    def test_convert_phyloxml_text(self):
        if False:
            return 10
        'Write phyloxml to a text handle.'
        trees = Phylo.parse('PhyloXML/phyloxml_examples.xml', 'phyloxml')
        with tempfile.NamedTemporaryFile(mode='w') as out_handle:
            count = Phylo.write(trees, out_handle, 'phyloxml')
        self.assertEqual(13, count)

    def test_convert_phyloxml_filename(self):
        if False:
            print('Hello World!')
        'Write phyloxml to a given filename.'
        trees = Phylo.parse('PhyloXML/phyloxml_examples.xml', 'phyloxml')
        out_handle = tempfile.NamedTemporaryFile(mode='w', delete=False)
        out_handle.close()
        tmp_filename = out_handle.name
        try:
            count = Phylo.write(trees, tmp_filename, 'phyloxml')
        finally:
            os.remove(tmp_filename)
        self.assertEqual(13, count)

    def test_int_labels(self):
        if False:
            for i in range(10):
                print('nop')
        'Read newick formatted tree with numeric labels.'
        tree = Phylo.read(StringIO('(((0:0.1,1:0.1)0.99:0.1,2:0.1)0.98:0.0);'), 'newick')
        self.assertEqual({leaf.name for leaf in tree.get_terminals()}, {'0', '1', '2'})

class TreeTests(unittest.TestCase):
    """Tests for methods on BaseTree.Tree objects."""

    def test_randomized(self):
        if False:
            for i in range(10):
                print('nop')
        'Tree.randomized: generate a new randomized tree.'
        for N in (2, 5, 20):
            tree = Phylo.BaseTree.Tree.randomized(N)
            self.assertEqual(tree.count_terminals(), N)
            self.assertEqual(tree.total_branch_length(), (N - 1) * 2)
            tree = Phylo.BaseTree.Tree.randomized(N, branch_length=2.0)
            self.assertEqual(tree.total_branch_length(), (N - 1) * 4)
        tree = Phylo.BaseTree.Tree.randomized(5, branch_stdev=0.5)
        self.assertEqual(tree.count_terminals(), 5)

    def test_root_with_outgroup(self):
        if False:
            for i in range(10):
                print('nop')
        'Tree.root_with_outgroup: reroot at a given clade.'
        tree = Phylo.read(EX_APAF, 'phyloxml')
        orig_num_tips = len(tree.get_terminals())
        orig_tree_len = tree.total_branch_length()
        tree.root_with_outgroup('19_NEMVE', '20_NEMVE')
        self.assertEqual(orig_num_tips, len(tree.get_terminals()))
        self.assertAlmostEqual(orig_tree_len, tree.total_branch_length())
        tree.root_with_outgroup('1_BRAFL')
        self.assertEqual(orig_num_tips, len(tree.get_terminals()))
        self.assertAlmostEqual(orig_tree_len, tree.total_branch_length())
        tree.root_with_outgroup('2_BRAFL', outgroup_branch_length=0.5)
        self.assertEqual(orig_num_tips, len(tree.get_terminals()))
        self.assertAlmostEqual(orig_tree_len, tree.total_branch_length())
        tree.root_with_outgroup('36_BRAFL', '37_BRAFL', outgroup_branch_length=0.5)
        self.assertEqual(orig_num_tips, len(tree.get_terminals()))
        self.assertAlmostEqual(orig_tree_len, tree.total_branch_length())
        for small_nwk in ('(A,B,(C,D));', '((E,F),((G,H)),(I,J));', '((Q,R),(S,T),(U,V));', '(X,Y);'):
            tree = Phylo.read(StringIO(small_nwk), 'newick')
            orig_tree_len = tree.total_branch_length()
            for node in list(tree.find_clades()):
                tree.root_with_outgroup(node)
                self.assertAlmostEqual(orig_tree_len, tree.total_branch_length())

    def test_root_at_midpoint(self):
        if False:
            i = 10
            return i + 15
        "Tree.root_at_midpoint: reroot at the tree's midpoint."
        for (treefname, fmt) in [(EX_APAF, 'phyloxml'), (EX_BCL2, 'phyloxml'), (EX_NEWICK, 'newick')]:
            tree = Phylo.read(treefname, fmt)
            orig_tree_len = tree.total_branch_length()
            tree.root_at_midpoint()
            self.assertAlmostEqual(orig_tree_len, tree.total_branch_length())
            self.assertEqual(len(tree.root.clades), 2)
            deep_dist_0 = max(tree.clade[0].depths().values())
            deep_dist_1 = max(tree.clade[1].depths().values())
            self.assertAlmostEqual(deep_dist_0, deep_dist_1)

    def test_str(self):
        if False:
            return 10
        'Tree.__str__: pretty-print to a string.\n\n        NB: The exact line counts are liable to change if the object\n        constructors change.\n        '
        for (source, count) in zip((EX_APAF, EX_BCL2, EX_DIST), (386, 747, 15)):
            tree = Phylo.read(source, 'phyloxml')
            output = str(tree)
            self.assertEqual(len(output.splitlines()), count)

class MixinTests(unittest.TestCase):
    """Tests for TreeMixin methods."""

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.phylogenies = list(Phylo.parse(EX_PHYLO, 'phyloxml'))

    def test_find_elements(self):
        if False:
            while True:
                i = 10
        'TreeMixin: find_elements() method.'
        tree = self.phylogenies[5]
        matches = list(tree.find_elements(PhyloXML.Taxonomy, code='OCTVU'))
        self.assertEqual(len(matches), 1)
        self.assertIsInstance(matches[0], PhyloXML.Taxonomy)
        self.assertEqual(matches[0].code, 'OCTVU')
        self.assertEqual(matches[0].scientific_name, 'Octopus vulgaris')
        tree = self.phylogenies[10]
        for (point, alt) in zip(tree.find_elements(geodetic_datum='WGS\\d{2}'), (472, 10, 452)):
            self.assertIsInstance(point, PhyloXML.Point)
            self.assertEqual(point.geodetic_datum, 'WGS84')
            self.assertAlmostEqual(point.alt, alt)
        tree = self.phylogenies[4]
        events = list(tree.find_elements(PhyloXML.Events))
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0].speciations, 1)
        self.assertEqual(events[1].duplications, 1)
        tree = self.phylogenies[3]
        taxonomy = tree.find_any('B. subtilis')
        self.assertEqual(taxonomy.scientific_name, 'B. subtilis')
        tree = Phylo.read(EX_APAF, 'phyloxml')
        domains = list(tree.find_elements(start=5))
        self.assertEqual(len(domains), 8)
        for dom in domains:
            self.assertEqual(dom.start, 5)
            self.assertEqual(dom.value, 'CARD')

    def test_find_clades(self):
        if False:
            print('Hello World!')
        'TreeMixin: find_clades() method.'
        for (clade, name) in zip(self.phylogenies[10].find_clades(name=True), list('ABCD')):
            self.assertIsInstance(clade, PhyloXML.Clade)
            self.assertEqual(clade.name, name)
        octo = list(self.phylogenies[5].find_clades(code='OCTVU'))
        self.assertEqual(len(octo), 1)
        self.assertIsInstance(octo[0], PhyloXML.Clade)
        self.assertEqual(octo[0].taxonomies[0].code, 'OCTVU')
        dee = next(self.phylogenies[10].find_clades('D'))
        self.assertEqual(dee.name, 'D')

    def test_find_terminal(self):
        if False:
            print('Hello World!')
        'TreeMixin: find_elements() with terminal argument.'
        for (tree, total, extern, intern) in zip(self.phylogenies, (6, 6, 7, 18, 21, 27, 7, 9, 9, 19, 15, 9, 6), (3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 3, 3), (3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3)):
            self.assertEqual(len(list(tree.find_elements())), total)
            self.assertEqual(len(list(tree.find_elements(terminal=True))), extern)
            self.assertEqual(len(list(tree.find_elements(terminal=False))), intern)

    def test_get_path(self):
        if False:
            while True:
                i = 10
        'TreeMixin: get_path() method.'
        path = self.phylogenies[1].get_path('B')
        self.assertEqual(len(path), 2)
        self.assertAlmostEqual(path[0].branch_length, 0.06)
        self.assertAlmostEqual(path[1].branch_length, 0.23)
        self.assertEqual(path[1].name, 'B')

    def test_trace(self):
        if False:
            return 10
        'TreeMixin: trace() method.'
        tree = self.phylogenies[1]
        path = tree.trace('A', 'C')
        self.assertEqual(len(path), 3)
        self.assertAlmostEqual(path[0].branch_length, 0.06)
        self.assertAlmostEqual(path[2].branch_length, 0.4)
        self.assertEqual(path[2].name, 'C')

    def test_common_ancestor(self):
        if False:
            i = 10
            return i + 15
        'TreeMixin: common_ancestor() method.'
        tree = self.phylogenies[1]
        lca = tree.common_ancestor('A', 'B')
        self.assertEqual(lca, tree.clade[0])
        lca = tree.common_ancestor('A', 'C')
        self.assertEqual(lca, tree.clade)
        tree = self.phylogenies[10]
        lca = tree.common_ancestor('A', 'B', 'C')
        self.assertEqual(lca, tree.clade[0])

    def test_depths(self):
        if False:
            print('Hello World!')
        'TreeMixin: depths() method.'
        tree = self.phylogenies[1]
        depths = tree.depths()
        self.assertEqual(len(depths), 5)
        for (found, expect) in zip(sorted(depths.values()), [0, 0.06, 0.162, 0.29, 0.4]):
            self.assertAlmostEqual(found, expect)

    def test_distance(self):
        if False:
            for i in range(10):
                print('nop')
        'TreeMixin: distance() method.'
        t = self.phylogenies[1]
        self.assertAlmostEqual(t.distance('A'), 0.162)
        self.assertAlmostEqual(t.distance('B'), 0.29)
        self.assertAlmostEqual(t.distance('C'), 0.4)
        self.assertAlmostEqual(t.distance('A', 'B'), 0.332)
        self.assertAlmostEqual(t.distance('A', 'C'), 0.562)
        self.assertAlmostEqual(t.distance('B', 'C'), 0.69)

    def test_is_bifurcating(self):
        if False:
            print('Hello World!')
        'TreeMixin: is_bifurcating() method.'
        for (tree, is_b) in zip(self.phylogenies, (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1)):
            self.assertEqual(tree.is_bifurcating(), is_b)

    def test_is_monophyletic(self):
        if False:
            while True:
                i = 10
        'TreeMixin: is_monophyletic() method.'
        tree = self.phylogenies[10]
        abcd = tree.get_terminals()
        abc = tree.clade[0].get_terminals()
        ab = abc[:2]
        d = tree.clade[1].get_terminals()
        self.assertEqual(tree.is_monophyletic(abcd), tree.root)
        self.assertEqual(tree.is_monophyletic(abc), tree.clade[0])
        self.assertFalse(tree.is_monophyletic(ab))
        self.assertEqual(tree.is_monophyletic(d), tree.clade[1])
        self.assertEqual(tree.is_monophyletic(*abcd), tree.root)

    def test_total_branch_length(self):
        if False:
            print('Hello World!')
        'TreeMixin: total_branch_length() method.'
        tree = self.phylogenies[1]
        self.assertAlmostEqual(tree.total_branch_length(), 0.792)
        self.assertAlmostEqual(tree.clade[0].total_branch_length(), 0.392)

    def test_collapse(self):
        if False:
            return 10
        'TreeMixin: collapse() method.'
        tree = self.phylogenies[1]
        parent = tree.collapse(tree.clade[0])
        self.assertEqual(len(parent), 3)
        for (clade, name, blength) in zip(parent, ('C', 'A', 'B'), (0.4, 0.162, 0.29)):
            self.assertEqual(clade.name, name)
            self.assertAlmostEqual(clade.branch_length, blength)

    def test_collapse_all(self):
        if False:
            return 10
        'TreeMixin: collapse_all() method.'
        tree = Phylo.read(EX_APAF, 'phyloxml')
        d1 = tree.depths()
        tree.collapse_all()
        d2 = tree.depths()
        for clade in d2:
            self.assertAlmostEqual(d1[clade], d2[clade])
        self.assertEqual(len(tree.get_terminals()), len(tree.clade))
        self.assertEqual(len(list(tree.find_clades(terminal=False))), 1)
        tree = Phylo.read(EX_APAF, 'phyloxml')
        d1 = tree.depths()
        internal_node_ct = len(tree.get_nonterminals())
        tree.collapse_all(lambda c: c.branch_length < 0.1)
        d2 = tree.depths()
        self.assertEqual(len(tree.get_nonterminals()), internal_node_ct - 7)
        for clade in d2:
            self.assertAlmostEqual(d1[clade], d2[clade])

    def test_ladderize(self):
        if False:
            print('Hello World!')
        'TreeMixin: ladderize() method.'

        def ordered_names(tree):
            if False:
                i = 10
                return i + 15
            return [n.name for n in tree.get_terminals()]
        tree = self.phylogenies[10]
        self.assertEqual(ordered_names(tree), list('ABCD'))
        tree.ladderize()
        self.assertEqual(ordered_names(tree), list('DABC'))
        tree.ladderize(reverse=True)
        self.assertEqual(ordered_names(tree), list('ABCD'))

    def test_prune(self):
        if False:
            for i in range(10):
                print('nop')
        'TreeMixin: prune() method.'
        tree = self.phylogenies[10]
        parent = tree.prune(name='B')
        self.assertEqual(len(parent.clades), 2)
        self.assertEqual(parent.clades[0].name, 'A')
        self.assertEqual(parent.clades[1].name, 'C')
        self.assertEqual(len(tree.get_terminals()), 3)
        self.assertEqual(len(tree.get_nonterminals()), 2)
        tree = self.phylogenies[0]
        parent = tree.prune(name='A')
        self.assertEqual(len(parent.clades), 2)
        for (clade, name, blen) in zip(parent, 'BC', (0.29, 0.4)):
            self.assertTrue(clade.is_terminal())
            self.assertEqual(clade.name, name)
            self.assertAlmostEqual(clade.branch_length, blen)
        self.assertEqual(len(tree.get_terminals()), 2)
        self.assertEqual(len(tree.get_nonterminals()), 1)
        tree = self.phylogenies[1]
        parent = tree.prune(name='C')
        self.assertEqual(parent, tree.root)
        self.assertEqual(len(parent.clades), 2)
        for (clade, name, blen) in zip(parent, 'AB', (0.102, 0.23)):
            self.assertTrue(clade.is_terminal())
            self.assertEqual(clade.name, name)
            self.assertAlmostEqual(clade.branch_length, blen)
        self.assertEqual(len(tree.get_terminals()), 2)
        self.assertEqual(len(tree.get_nonterminals()), 1)

    def test_split(self):
        if False:
            print('Hello World!')
        'TreeMixin: split() method.'
        tree = self.phylogenies[0]
        C = tree.clade[1]
        C.split()
        self.assertEqual(len(C), 2)
        self.assertEqual(len(tree.get_terminals()), 4)
        self.assertEqual(len(tree.get_nonterminals()), 3)
        C[0].split(3, 0.5)
        self.assertEqual(len(tree.get_terminals()), 6)
        self.assertEqual(len(tree.get_nonterminals()), 4)
        for (clade, name, blen) in zip(C[0], ('C00', 'C01', 'C02'), (0.5, 0.5, 0.5)):
            self.assertTrue(clade.is_terminal())
            self.assertEqual(clade.name, name)
            self.assertEqual(clade.branch_length, blen)
if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    unittest.main(testRunner=runner)