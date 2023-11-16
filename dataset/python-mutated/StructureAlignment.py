"""Map residues of two structures to each other based on a FASTA alignment."""
from Bio.Data import PDBData
from Bio.PDB import Selection
from Bio.PDB.Polypeptide import is_aa

class StructureAlignment:
    """Class to align two structures based on an alignment of their sequences."""

    def __init__(self, fasta_align, m1, m2, si=0, sj=1):
        if False:
            print('Hello World!')
        'Initialize.\n\n        Attributes:\n         - fasta_align - Alignment object\n         - m1, m2 - two models\n         - si, sj - the sequences in the Alignment object that\n           correspond to the structures\n\n        '
        try:
            ncolumns = fasta_align.get_alignment_length()
        except AttributeError:
            (nrows, ncolumns) = fasta_align.shape
        rl1 = Selection.unfold_entities(m1, 'R')
        rl2 = Selection.unfold_entities(m2, 'R')
        p1 = 0
        p2 = 0
        map12 = {}
        map21 = {}
        duos = []
        for i in range(ncolumns):
            column = fasta_align[:, i]
            aa1 = column[si]
            aa2 = column[sj]
            if aa1 != '-':
                while True:
                    r1 = rl1[p1]
                    p1 = p1 + 1
                    if is_aa(r1):
                        break
                self._test_equivalence(r1, aa1)
            else:
                r1 = None
            if aa2 != '-':
                while True:
                    r2 = rl2[p2]
                    p2 = p2 + 1
                    if is_aa(r2):
                        break
                self._test_equivalence(r2, aa2)
            else:
                r2 = None
            if r1:
                map12[r1] = r2
            if r2:
                map21[r2] = r1
            duos.append((r1, r2))
        self.map12 = map12
        self.map21 = map21
        self.duos = duos

    def _test_equivalence(self, r1, aa1):
        if False:
            while True:
                i = 10
        'Test if aa in sequence fits aa in structure (PRIVATE).'
        resname = r1.get_resname()
        resname = PDBData.protein_letters_3to1_extended[resname]
        assert aa1 == resname

    def get_maps(self):
        if False:
            i = 10
            return i + 15
        'Map residues between the structures.\n\n        Return two dictionaries that map a residue in one structure to\n        the equivealent residue in the other structure.\n        '
        return (self.map12, self.map21)

    def get_iterator(self):
        if False:
            return 10
        'Create an iterator over all residue pairs.'
        for i in range(len(self.duos)):
            yield self.duos[i]