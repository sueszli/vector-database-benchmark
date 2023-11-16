"""Fast atom neighbor lookup using a KD tree (implemented in C)."""
import numpy as np
from Bio.PDB.PDBExceptions import PDBException
from Bio.PDB.Selection import unfold_entities, entity_levels, uniqueify

class NeighborSearch:
    """Class for neighbor searching.

    This class can be used for two related purposes:

     1. To find all atoms/residues/chains/models/structures within radius
        of a given query position.
     2. To find all atoms/residues/chains/models/structures that are within
        a fixed radius of each other.

    NeighborSearch makes use of the KDTree class implemented in C for speed.
    """

    def __init__(self, atom_list, bucket_size=10):
        if False:
            i = 10
            return i + 15
        'Create the object.\n\n        Arguments:\n         - atom_list - list of atoms. This list is used in the queries.\n           It can contain atoms from different structures.\n         - bucket_size - bucket size of KD tree. You can play around\n           with this to optimize speed if you feel like it.\n\n        '
        from Bio.PDB.kdtrees import KDTree
        self.atom_list = atom_list
        coord_list = [a.get_coord() for a in atom_list]
        self.coords = np.array(coord_list, dtype='d')
        assert bucket_size > 1
        assert self.coords.shape[1] == 3
        self.kdt = KDTree(self.coords, bucket_size)

    def _get_unique_parent_pairs(self, pair_list):
        if False:
            while True:
                i = 10
        parent_pair_list = []
        for (e1, e2) in pair_list:
            p1 = e1.get_parent()
            p2 = e2.get_parent()
            if p1 == p2:
                continue
            elif p1 < p2:
                parent_pair_list.append((p1, p2))
            else:
                parent_pair_list.append((p2, p1))
        return uniqueify(parent_pair_list)

    def search(self, center, radius, level='A'):
        if False:
            while True:
                i = 10
        'Neighbor search.\n\n        Return all atoms/residues/chains/models/structures\n        that have at least one atom within radius of center.\n        What entity level is returned (e.g. atoms or residues)\n        is determined by level (A=atoms, R=residues, C=chains,\n        M=models, S=structures).\n\n        Arguments:\n         - center - NumPy array\n         - radius - float\n         - level - char (A, R, C, M, S)\n\n        '
        if level not in entity_levels:
            raise PDBException(f'{level}: Unknown level')
        center = np.require(center, dtype='d', requirements='C')
        if center.shape != (3,):
            raise Exception('Expected a 3-dimensional NumPy array')
        points = self.kdt.search(center, radius)
        atom_list = [self.atom_list[point.index] for point in points]
        if level == 'A':
            return atom_list
        else:
            return unfold_entities(atom_list, level)

    def search_all(self, radius, level='A'):
        if False:
            i = 10
            return i + 15
        'All neighbor search.\n\n        Search all entities that have atoms pairs within\n        radius.\n\n        Arguments:\n         - radius - float\n         - level - char (A, R, C, M, S)\n\n        '
        if level not in entity_levels:
            raise PDBException(f'{level}: Unknown level')
        neighbors = self.kdt.neighbor_search(radius)
        atom_list = self.atom_list
        atom_pair_list = []
        for neighbor in neighbors:
            i1 = neighbor.index1
            i2 = neighbor.index2
            a1 = atom_list[i1]
            a2 = atom_list[i2]
            atom_pair_list.append((a1, a2))
        if level == 'A':
            return atom_pair_list
        next_level_pair_list = atom_pair_list
        for next_level in ['R', 'C', 'M', 'S']:
            next_level_pair_list = self._get_unique_parent_pairs(next_level_pair_list)
            if level == next_level:
                return next_level_pair_list