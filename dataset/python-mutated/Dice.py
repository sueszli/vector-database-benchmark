"""Code for chopping up (dicing) a structure.

This module is used internally by the Bio.PDB.extract() function.
"""
import re
import warnings
from Bio.PDB.PDBIO import PDBIO
from Bio import BiopythonWarning
_hydrogen = re.compile('[123 ]*H.*')

class ChainSelector:
    """Only accepts residues with right chainid, between start and end.

    Remove hydrogens, waters and ligands. Only use model 0 by default.
    """

    def __init__(self, chain_id, start, end, model_id=0):
        if False:
            i = 10
            return i + 15
        'Initialize the class.'
        self.chain_id = chain_id
        self.start = start
        self.end = end
        self.model_id = model_id

    def accept_model(self, model):
        if False:
            print('Hello World!')
        'Verify if model match the model identifier.'
        if model.get_id() == self.model_id:
            return 1
        return 0

    def accept_chain(self, chain):
        if False:
            while True:
                i = 10
        'Verify if chain match chain identifier.'
        if chain.get_id() == self.chain_id:
            return 1
        return 0

    def accept_residue(self, residue):
        if False:
            for i in range(10):
                print('nop')
        'Verify if a residue sequence is between the start and end sequence.'
        (hetatm_flag, resseq, icode) = residue.get_id()
        if hetatm_flag != ' ':
            return 0
        if icode != ' ':
            warnings.warn(f'WARNING: Icode {icode} at position {resseq}', BiopythonWarning)
        if self.start <= resseq <= self.end:
            return 1
        return 0

    def accept_atom(self, atom):
        if False:
            return 10
        'Verify if atoms are not Hydrogen.'
        name = atom.get_id()
        if _hydrogen.match(name):
            return 0
        else:
            return 1

def extract(structure, chain_id, start, end, filename):
    if False:
        return 10
    'Write out selected portion to filename.'
    sel = ChainSelector(chain_id, start, end)
    io = PDBIO()
    io.set_structure(structure)
    io.save(filename, sel)