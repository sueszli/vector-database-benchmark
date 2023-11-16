"""Tests for SeqIO PdbIO module."""
import unittest
import warnings
try:
    import numpy
    from numpy import dot
    del dot
    from numpy.linalg import svd, det
except ImportError:
    from Bio import MissingPythonDependencyError
    raise MissingPythonDependencyError('Install NumPy if you want to use PDB formats with SeqIO.') from None
from Bio import SeqIO
from Bio import BiopythonParserWarning
from Bio.PDB.PDBExceptions import PDBConstructionWarning

def SeqresTestGenerator(extension, parser):
    if False:
        print('Hello World!')
    'Test factory for tests reading SEQRES (or similar) records.\n\n    This is a factory returning a parameterised superclass for tests reading\n    sequences from the sequence records of structure files.\n\n    Arguments:\n        extension:\n            The extension of the files to read from the ``PDB`` directory (e.g.\n            ``pdb`` or ``cif``).\n        parser:\n            The name of the SeqIO parser to use (e.g. ``pdb-atom``).\n\n    '

    class SeqresTests(unittest.TestCase):
        """Use "parser" to parse sequence records from a structure file.

        Args:
            parser (str): Name of the parser used by SeqIO.
            extension (str): Extension of the files to parse.

        """

        def test_seqres_parse(self):
            if False:
                while True:
                    i = 10
            'Parse a multi-chain PDB by SEQRES entries.\n\n            Reference:\n            http://www.rcsb.org/pdb/files/fasta.txt?structureIdList=2BEG\n            '
            chains = list(SeqIO.parse('PDB/2BEG.' + extension, parser))
            self.assertEqual(len(chains), 5)
            actual_seq = 'DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA'
            for (chain, chn_id) in zip(chains, 'ABCDE'):
                self.assertEqual(chain.id, '2BEG:' + chn_id)
                self.assertEqual(chain.annotations['chain'], chn_id)
                self.assertEqual(chain.seq, actual_seq)

        def test_seqres_read(self):
            if False:
                return 10
            'Read a single-chain structure by sequence entries.\n\n            Reference:\n            http://www.rcsb.org/pdb/files/fasta.txt?structureIdList=1A8O\n            '
            chain = SeqIO.read('PDB/1A8O.' + extension, parser)
            self.assertEqual(chain.id, '1A8O:A')
            self.assertEqual(chain.annotations['chain'], 'A')
            self.assertEqual(chain.seq, 'MDIRQGPKEPFRDYVDRFYKTLRAEQASQEVKNWMTETLLVQNANPDCKTILKALGPGATLEEMMTACQG')

        def test_seqres_missing(self):
            if False:
                print('Hello World!')
            'Parse a PDB with no SEQRES entries.'
            chains = list(SeqIO.parse('PDB/a_structure.' + extension, parser))
            self.assertEqual(len(chains), 0)
    return SeqresTests

class TestPdbSeqres(SeqresTestGenerator('pdb', 'pdb-seqres')):
    """Test pdb-seqres SeqIO driver."""

class TestCifSeqres(SeqresTestGenerator('cif', 'cif-seqres')):
    """Test cif-seqres SeqIO driver."""

def AtomTestGenerator(extension, parser):
    if False:
        return 10
    'Test factory for tests reading ATOM (or similar) records.\n\n    See SeqresTestGenerator for more information.\n    '

    class AtomTests(unittest.TestCase):

        def test_atom_parse(self):
            if False:
                print('Hello World!')
            'Parse a multi-chain structure by ATOM entries.\n\n            Reference:\n            http://www.rcsb.org/pdb/files/fasta.txt?structureIdList=2BEG\n            '
            chains = list(SeqIO.parse('PDB/2BEG.' + extension, parser))
            self.assertEqual(len(chains), 5)
            actual_seq = 'LVFFAEDVGSNKGAIIGLMVGGVVIA'
            for (chain, chn_id) in zip(chains, 'ABCDE'):
                self.assertEqual(chain.id, '2BEG:' + chn_id)
                self.assertEqual(chain.annotations['chain'], chn_id)
                self.assertEqual(chain.annotations['model'], 0)
                self.assertEqual(chain.seq, actual_seq)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', PDBConstructionWarning)
                chains = list(SeqIO.parse('PDB/2XHE.' + extension, parser))
            actual_seq = 'DRLSRLRQMAAENQXXXXXXXXXXXXXXXXXXXXXXXPEPFMADFFNRVKRIRDNIEDIEQAIEQVAQLHTESLVAVSKEDRDRLNEKLQDTMARISALGNKIRADLKQIEKENKRAQQEGTFEDGTVSTDLRIRQSQHSSLSRKFVKVMTRYNDVQAENKRRYGENVARQCRVVEPSLSDDAIQKVIEHGXXXXXXXXXXXXXXXXXNEIRDRHKDIQQLERSLLELHEMFTDMSTLVASQGEMIDRIEFSVEQSHNYV'
            self.assertEqual(chains[1].seq, actual_seq)

        def test_atom_read(self):
            if False:
                i = 10
                return i + 15
            'Read a single-chain structure by ATOM entries.\n\n            Reference:\n            http://www.rcsb.org/pdb/files/fasta.txt?structureIdList=1A8O\n            '
            chain = SeqIO.read('PDB/1A8O.' + extension, parser)
            self.assertEqual(chain.id, '1A8O:A')
            self.assertEqual(chain.annotations['chain'], 'A')
            self.assertEqual(chain.annotations['model'], 0)
            self.assertEqual(chain.seq, 'MDIRQGPKEPFRDYVDRFYKTLRAEQASQEVKNWMTETLLVQNANPDCKTILKALGPGATLEEMMTACQG')
    return AtomTests

class TestPdbAtom(AtomTestGenerator('pdb', 'pdb-atom')):
    """Test pdb-atom SeqIO driver."""

    def test_atom_noheader(self):
        if False:
            return 10
        'Parse a PDB with no HEADER line.'
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', PDBConstructionWarning)
            warnings.simplefilter('ignore', BiopythonParserWarning)
            chains = list(SeqIO.parse('PDB/1LCD.pdb', 'pdb-atom'))
        self.assertEqual(len(chains), 1)
        self.assertEqual(chains[0].seq, 'MKPVTLYDVAEYAGVSYQTVSRVVNQASHVSAKTREKVEAAMAELNYIPNR')

    def test_atom_read_noheader(self):
        if False:
            i = 10
            return i + 15
        'Read a single-chain PDB without a header by ATOM entries.'
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', PDBConstructionWarning)
            warnings.simplefilter('ignore', BiopythonParserWarning)
            chain = SeqIO.read('PDB/a_structure.pdb', 'pdb-atom')
        self.assertEqual(chain.id, '????:A')
        self.assertEqual(chain.annotations['chain'], 'A')
        self.assertEqual(chain.seq, 'Q')

    def test_atom_with_insertion(self):
        if False:
            print('Hello World!')
        'Read a PDB with residue insertion code.'
        chain = SeqIO.read('PDB/2n0n_M1.pdb', 'pdb-atom')
        self.assertEqual(chain.seq, 'HAEGKFTSEF')

class TestCifAtom(AtomTestGenerator('cif', 'cif-atom')):
    """Test cif-atom SeqIO driver."""

    def test_atom_read_noheader(self):
        if False:
            return 10
        'Read a single-chain CIF without a header by ATOM entries.'
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', PDBConstructionWarning)
            warnings.simplefilter('ignore', BiopythonParserWarning)
            chain = SeqIO.read('PDB/a_structure.cif', 'cif-atom')
        self.assertEqual(chain.id, '????:A')
        self.assertEqual(chain.annotations['chain'], 'A')
        self.assertEqual(chain.seq, 'MDIRQGPKEPFRDYVDRFYKTLRAEQASQEVKNWMTETLLVQNANPDCKTILKALGPGATLEEMMTACQG')
if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    unittest.main(testRunner=runner)