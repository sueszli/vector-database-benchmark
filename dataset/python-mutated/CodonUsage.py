"""Methods for codon usage calculations."""
import math
import warnings
from .CodonUsageIndices import SharpEcoliIndex
from Bio import SeqIO
from Bio import BiopythonDeprecationWarning
warnings.warn('This module has been DEPRECATED. Please use the CodonAdaptationIndex class in Bio.SeqUtils instead. Note that this class has been updated to use modern Python, and may give slightly different results from the CodonAdaptationIndex class in Bio.SeqUtils.CodonUsage, as the code was changed to be consistent with the published paper by Sharp and Li. The code in the old CodonAdaptationIndex class in Bio.SeqUtils.CodonUsage was not changed.', BiopythonDeprecationWarning)
CodonsDict = {'TTT': 0, 'TTC': 0, 'TTA': 0, 'TTG': 0, 'CTT': 0, 'CTC': 0, 'CTA': 0, 'CTG': 0, 'ATT': 0, 'ATC': 0, 'ATA': 0, 'ATG': 0, 'GTT': 0, 'GTC': 0, 'GTA': 0, 'GTG': 0, 'TAT': 0, 'TAC': 0, 'TAA': 0, 'TAG': 0, 'CAT': 0, 'CAC': 0, 'CAA': 0, 'CAG': 0, 'AAT': 0, 'AAC': 0, 'AAA': 0, 'AAG': 0, 'GAT': 0, 'GAC': 0, 'GAA': 0, 'GAG': 0, 'TCT': 0, 'TCC': 0, 'TCA': 0, 'TCG': 0, 'CCT': 0, 'CCC': 0, 'CCA': 0, 'CCG': 0, 'ACT': 0, 'ACC': 0, 'ACA': 0, 'ACG': 0, 'GCT': 0, 'GCC': 0, 'GCA': 0, 'GCG': 0, 'TGT': 0, 'TGC': 0, 'TGA': 0, 'TGG': 0, 'CGT': 0, 'CGC': 0, 'CGA': 0, 'CGG': 0, 'AGT': 0, 'AGC': 0, 'AGA': 0, 'AGG': 0, 'GGT': 0, 'GGC': 0, 'GGA': 0, 'GGG': 0}
SynonymousCodons = {'CYS': ['TGT', 'TGC'], 'ASP': ['GAT', 'GAC'], 'SER': ['TCT', 'TCG', 'TCA', 'TCC', 'AGC', 'AGT'], 'GLN': ['CAA', 'CAG'], 'MET': ['ATG'], 'ASN': ['AAC', 'AAT'], 'PRO': ['CCT', 'CCG', 'CCA', 'CCC'], 'LYS': ['AAG', 'AAA'], 'STOP': ['TAG', 'TGA', 'TAA'], 'THR': ['ACC', 'ACA', 'ACG', 'ACT'], 'PHE': ['TTT', 'TTC'], 'ALA': ['GCA', 'GCC', 'GCG', 'GCT'], 'GLY': ['GGT', 'GGG', 'GGA', 'GGC'], 'ILE': ['ATC', 'ATA', 'ATT'], 'LEU': ['TTA', 'TTG', 'CTC', 'CTT', 'CTG', 'CTA'], 'HIS': ['CAT', 'CAC'], 'ARG': ['CGA', 'CGC', 'CGG', 'CGT', 'AGG', 'AGA'], 'TRP': ['TGG'], 'VAL': ['GTA', 'GTC', 'GTG', 'GTT'], 'GLU': ['GAG', 'GAA'], 'TYR': ['TAT', 'TAC']}

class CodonAdaptationIndex:
    """A codon adaptation index (CAI) implementation.

    Implements the codon adaptation index (CAI) described by Sharp and
    Li (Nucleic Acids Res. 1987 Feb 11;15(3):1281-95).

    NOTE - This implementation does not currently cope with alternative genetic
    codes: only the synonymous codons in the standard table are considered.
    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        'Initialize the class.'
        self.index = {}
        self.codon_count = {}

    def set_cai_index(self, index):
        if False:
            while True:
                i = 10
        'Set up an index to be used when calculating CAI for a gene.\n\n        Just pass a dictionary similar to the SharpEcoliIndex in the\n        CodonUsageIndices module.\n        '
        self.index = index

    def generate_index(self, fasta_file):
        if False:
            return 10
        'Generate a codon usage index from a FASTA file of CDS sequences.\n\n        Takes a location of a Fasta file containing CDS sequences\n        (which must all have a whole number of codons) and generates a codon\n        usage index.\n        '
        if self.index != {} or self.codon_count != {}:
            raise ValueError('an index has already been set or a codon count has been done. Cannot overwrite either.')
        self._count_codons(fasta_file)
        for aa in SynonymousCodons:
            codons = SynonymousCodons[aa]
            count_max = max((self.codon_count[codon] for codon in codons))
            if count_max == 0:
                for codon in codons:
                    self.index[codon] = None
            else:
                for codon in codons:
                    self.index[codon] = self.codon_count[codon] / count_max

    def cai_for_gene(self, dna_sequence):
        if False:
            print('Hello World!')
        'Calculate the CAI (float) for the provided DNA sequence (string).\n\n        This method uses the Index (either the one you set or the one you\n        generated) and returns the CAI for the DNA sequence.\n        '
        (cai_value, cai_length) = (0, 0)
        if self.index == {}:
            self.set_cai_index(SharpEcoliIndex)
        dna_sequence = dna_sequence.upper()
        for i in range(0, len(dna_sequence), 3):
            codon = dna_sequence[i:i + 3]
            if codon in self.index:
                if codon not in ['ATG', 'TGG']:
                    cai_value += math.log(self.index[codon])
                    cai_length += 1
            elif codon not in ['TGA', 'TAA', 'TAG']:
                raise TypeError(f'illegal codon in sequence: {codon}.\n{self.index}')
        return math.exp(cai_value / (cai_length - 1.0))

    def _count_codons(self, fasta_file):
        if False:
            print('Hello World!')
        with open(fasta_file) as handle:
            self.codon_count = CodonsDict.copy()
            for record in SeqIO.parse(handle, 'fasta'):
                sequence = record.seq.upper()
                for i in range(0, len(sequence), 3):
                    codon = sequence[i:i + 3]
                    try:
                        self.codon_count[codon] += 1
                    except KeyError:
                        raise ValueError(f"illegal codon '{codon}' in gene: {record.id}") from None

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        lines = []
        for i in sorted(self.index):
            line = f'{i}\t{self.index[i]:.3f}'
            lines.append(line)
        return '\n'.join(lines) + '\n'

    def print_index(self):
        if False:
            print('Hello World!')
        'Print out the index used.\n\n        This just gives the index when the objects is printed.\n        '
        warnings.warn('The print_index method is deprecated; instead of self.print_index(), please use print(self).', BiopythonDeprecationWarning)
        print(self)