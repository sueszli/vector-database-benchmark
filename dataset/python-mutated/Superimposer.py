"""Superimpose two structures."""
import numpy as np
from Bio.SVDSuperimposer import SVDSuperimposer
from Bio.PDB.PDBExceptions import PDBException

class Superimposer:
    """Rotate/translate one set of atoms on top of another to minimize RMSD."""

    def __init__(self):
        if False:
            while True:
                i = 10
        'Initialize the class.'
        self.rotran = None
        self.rms = None

    def set_atoms(self, fixed, moving):
        if False:
            i = 10
            return i + 15
        'Prepare translation/rotation to minimize RMSD between atoms.\n\n        Put (translate/rotate) the atoms in fixed on the atoms in\n        moving, in such a way that the RMSD is minimized.\n\n        :param fixed: list of (fixed) atoms\n        :param moving: list of (moving) atoms\n        :type fixed,moving: [L{Atom}, L{Atom},...]\n        '
        if not len(fixed) == len(moving):
            raise PDBException('Fixed and moving atom lists differ in size')
        length = len(fixed)
        fixed_coord = np.zeros((length, 3))
        moving_coord = np.zeros((length, 3))
        for i in range(length):
            fixed_coord[i] = fixed[i].get_coord()
            moving_coord[i] = moving[i].get_coord()
        sup = SVDSuperimposer()
        sup.set(fixed_coord, moving_coord)
        sup.run()
        self.rms = sup.get_rms()
        self.rotran = sup.get_rotran()

    def apply(self, atom_list):
        if False:
            print('Hello World!')
        'Rotate/translate a list of atoms.'
        if self.rotran is None:
            raise PDBException('No transformation has been calculated yet')
        (rot, tran) = self.rotran
        rot = rot.astype('f')
        tran = tran.astype('f')
        for atom in atom_list:
            atom.transform(rot, tran)