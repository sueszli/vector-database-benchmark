"""Tests for EmbossPhylipNew module."""
import os
import sys
import unittest
from Bio import MissingExternalDependencyError
from Bio import AlignIO
from Bio.Nexus import Trees
from Bio.Emboss.Applications import FDNADistCommandline, FNeighborCommandline
from Bio.Emboss.Applications import FSeqBootCommandline, FProtDistCommandline
from Bio.Emboss.Applications import FProtParsCommandline, FConsenseCommandline
from Bio.Emboss.Applications import FTreeDistCommandline, FDNAParsCommandline
os.environ['LANG'] = 'C'
exes_wanted = ['fdnadist', 'fneighbor', 'fprotdist', 'fprotpars', 'fconsense', 'fseqboot', 'ftreedist', 'fdnapars']
exes = {}
if 'EMBOSS_ROOT' in os.environ:
    path = os.environ['EMBOSS_ROOT']
    if os.path.isdir(path):
        for name in exes_wanted:
            if os.path.isfile(os.path.join(path, name + '.exe')):
                exes[name] = os.path.join(path, name + '.exe')
    del path, name
if sys.platform != 'win32':
    from subprocess import getoutput
    for name in exes_wanted:
        output = getoutput(f'{name} -help')
        if 'not found' not in output and 'not recognized' not in output:
            exes[name] = name
        del output
    del name
if len(exes) < len(exes_wanted):
    raise MissingExternalDependencyError("Install the Emboss package 'PhylipNew' if you want to use the Bio.Emboss.Applications wrappers for phylogenetic tools.")

def clean_up():
    if False:
        while True:
            i = 10
    'Delete tests files (to be used as tearDown() function in test fixtures).'
    for filename in ['test_file', 'Phylip/opuntia.phy', 'Phylip/hedgehog.phy']:
        if os.path.isfile(filename):
            os.remove(filename)

def parse_trees(filename):
    if False:
        i = 10
        return i + 15
    'Parse trees.\n\n    Helper function until we have Bio.Phylo on trunk.\n    '
    with open('test_file') as handle:
        data = handle.read()
    for tree_str in data.split(';\n'):
        if tree_str:
            yield Trees.Tree(tree_str + ';')

class DistanceTests(unittest.TestCase):
    """Tests for calculating distance based phylogenetic trees with phylip."""

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        clean_up()
    test_taxa = ['Archaeohip', 'Calippus', 'Hypohippus', 'M._secundu', 'Merychippu', 'Mesohippus', 'Nannipus', 'Neohippari', 'Parahippus', 'Pliohippus']

    def distances_from_alignment(self, filename, DNA=True):
        if False:
            i = 10
            return i + 15
        'Check we can make a distance matrix from a given alignment.'
        self.assertTrue(os.path.isfile(filename), f'Missing {filename}')
        if DNA:
            cline = FDNADistCommandline(exes['fdnadist'], method='j', sequence=filename, outfile='test_file', auto=True)
        else:
            cline = FProtDistCommandline(exes['fprotdist'], method='j', sequence=filename, outfile='test_file', auto=True)
        (stdout, strerr) = cline()
        self.assertTrue(os.path.isfile('test_file'))

    def tree_from_distances(self, filename):
        if False:
            return 10
        'Check we can estimate a tree from a distance matrix.'
        self.assertTrue(os.path.isfile(filename), f'Missing {filename}')
        cline = FNeighborCommandline(exes['fneighbor'], datafile=filename, outtreefile='test_file', auto=True, filter=True)
        (stdout, stderr) = cline()
        for tree in parse_trees('test_file'):
            tree_taxa = [t.replace(' ', '_') for t in tree.get_taxa()]
            self.assertEqual(self.test_taxa, sorted(tree_taxa))

    def test_distances_from_phylip_DNA(self):
        if False:
            return 10
        'Calculate a distance matrix from an phylip alignment.'
        self.distances_from_alignment('Phylip/horses.phy')

    def test_distances_from_AlignIO_DNA(self):
        if False:
            while True:
                i = 10
        'Calculate a distance matrix from an alignment written by AlignIO.'
        n = AlignIO.convert('Clustalw/opuntia.aln', 'clustal', 'Phylip/opuntia.phy', 'phylip')
        self.assertEqual(n, 1)
        self.distances_from_alignment('Phylip/opuntia.phy')

    def test_distances_from_protein_phylip(self):
        if False:
            for i in range(10):
                print('nop')
        'Calculate a distance matrix from phylip protein alignment.'
        self.distances_from_alignment('Phylip/interlaced.phy', DNA=False)

    def test_distances_from_protein_AlignIO(self):
        if False:
            for i in range(10):
                print('nop')
        'Calculate distance matrix from an AlignIO written protein alignment.'
        n = AlignIO.convert('Clustalw/hedgehog.aln', 'clustal', 'Phylip/hedgehog.phy', 'phylip')
        self.assertEqual(n, 1)
        self.distances_from_alignment('Phylip/hedgehog.phy', DNA=False)

class ParsimonyTests(unittest.TestCase):
    """Tests for estimating parsimony based phylogenetic trees with phylip."""

    def tearDown(self):
        if False:
            return 10
        clean_up()

    def parsimony_tree(self, filename, format, DNA=True):
        if False:
            print('Hello World!')
        'Estimate a parsimony tree from an alignment.'
        self.assertTrue(os.path.isfile(filename), f'Missing {filename}')
        if DNA:
            cline = FDNAParsCommandline(exes['fdnapars'], sequence=filename, outtreefile='test_file', auto=True, stdout=True)
        else:
            cline = FProtParsCommandline(exes['fprotpars'], sequence=filename, outtreefile='test_file', auto=True, stdout=True)
        (stdout, stderr) = cline()
        with open(filename) as handle:
            a_taxa = [s.name.replace(' ', '_') for s in next(AlignIO.parse(handle, format))]
        for tree in parse_trees('test_file'):
            t_taxa = [t.replace(' ', '_') for t in tree.get_taxa()]
            self.assertEqual(sorted(a_taxa), sorted(t_taxa))

    def test_parsimony_tree_from_AlignIO_DNA(self):
        if False:
            for i in range(10):
                print('nop')
        'Make a parsimony tree from an alignment written with AlignIO.'
        n = AlignIO.convert('Clustalw/opuntia.aln', 'clustal', 'Phylip/opuntia.phy', 'phylip')
        self.assertEqual(n, 1)
        self.parsimony_tree('Phylip/opuntia.phy', 'phylip')

    def test_parsimony_from_AlignIO_protein(self):
        if False:
            i = 10
            return i + 15
        'Make a parsimony tree from protein alignment written with AlignIO.'
        n = AlignIO.convert('Clustalw/hedgehog.aln', 'clustal', 'Phylip/hedgehog.phy', 'phylip')
        self.parsimony_tree('Phylip/interlaced.phy', 'phylip', DNA=False)

class BootstrapTests(unittest.TestCase):
    """Tests for pseudosampling alignments with fseqboot."""

    def tearDown(self):
        if False:
            print('Hello World!')
        clean_up()

    def check_bootstrap(self, filename, format, align_type='d'):
        if False:
            while True:
                i = 10
        'Check we can use fseqboot to pseudosample an alignment.\n\n        The align_type type argument is passed to the commandline object to\n        set the output format to use (from [D]na,[p]rotein and [r]na )\n        '
        self.assertTrue(os.path.isfile(filename), f'Missing {filename}')
        cline = FSeqBootCommandline(exes['fseqboot'], sequence=filename, outfile='test_file', seqtype=align_type, reps=2, auto=True, filter=True)
        (stdout, stderr) = cline()
        with open('test_file') as handle:
            bs = list(AlignIO.parse(handle, format))
        self.assertEqual(len(bs), 2)
        with open(filename) as handle:
            a_names = [s.name.replace(' ', '_') for s in AlignIO.read(handle, format)]
        for a in bs:
            self.assertEqual(a_names, [s.name.replace(' ', '_') for s in a])

    def test_bootstrap_phylip_DNA(self):
        if False:
            print('Hello World!')
        'Pseudosample a phylip DNA alignment.'
        self.check_bootstrap('Phylip/horses.phy', 'phylip')

    def test_bootstrap_AlignIO_DNA(self):
        if False:
            return 10
        'Pseudosample a phylip DNA alignment written with AlignIO.'
        n = AlignIO.convert('Clustalw/opuntia.aln', 'clustal', 'Phylip/opuntia.phy', 'phylip')
        self.assertEqual(n, 1)
        self.check_bootstrap('Phylip/opuntia.phy', 'phylip')

    def test_bootstrap_phylip_protein(self):
        if False:
            i = 10
            return i + 15
        'Pseudosample a phylip protein alignment.'
        self.check_bootstrap('Phylip/interlaced.phy', 'phylip', 'p')

    def test_bootstrap_AlignIO_protein(self):
        if False:
            while True:
                i = 10
        'Pseudosample a phylip protein alignment written with AlignIO.'
        n = AlignIO.convert('Clustalw/hedgehog.aln', 'clustal', 'Phylip/hedgehog.phy', 'phylip')
        self.check_bootstrap('Phylip/hedgehog.phy', 'phylip', 'p')

class TreeComparisonTests(unittest.TestCase):
    """Tests for comparing phylogenetic trees with phylip tools."""

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        clean_up()

    def test_fconsense(self):
        if False:
            print('Hello World!')
        'Calculate a consensus tree with fconsense.'
        cline = FConsenseCommandline(exes['fconsense'], intreefile='Phylip/horses.tree', outtreefile='test_file', auto=True, filter=True)
        (stdout, stderr) = cline()
        tree1 = next(parse_trees('test_file'))
        taxa1 = tree1.get_taxa()
        for tree in parse_trees('Phylip/horses.tree'):
            taxa2 = tree.get_taxa()
            self.assertEqual(sorted(taxa1), sorted(taxa2))

    def test_ftreedist(self):
        if False:
            return 10
        'Calculate the distance between trees with ftreedist.'
        cline = FTreeDistCommandline(exes['ftreedist'], intreefile='Phylip/horses.tree', outfile='test_file', auto=True, filter=True)
        (stdout, stderr) = cline()
        self.assertTrue(os.path.isfile('test_file'))
if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    unittest.main(testRunner=runner)
    clean_up()