"""The structure class, representing a macromolecular structure."""
from Bio.PDB.Entity import Entity

class Structure(Entity):
    """The Structure class contains a collection of Model instances."""

    def __init__(self, id):
        if False:
            while True:
                i = 10
        'Initialize the class.'
        self.level = 'S'
        Entity.__init__(self, id)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        'Return the structure identifier.'
        return f'<Structure id={self.get_id()}>'

    def get_models(self):
        if False:
            for i in range(10):
                print('nop')
        'Return models.'
        yield from self

    def get_chains(self):
        if False:
            return 10
        'Return chains from models.'
        for m in self.get_models():
            yield from m

    def get_residues(self):
        if False:
            i = 10
            return i + 15
        'Return residues from chains.'
        for c in self.get_chains():
            yield from c

    def get_atoms(self):
        if False:
            i = 10
            return i + 15
        'Return atoms from residue.'
        for r in self.get_residues():
            yield from r

    def atom_to_internal_coordinates(self, verbose: bool=False) -> None:
        if False:
            return 10
        'Create/update internal coordinates from Atom X,Y,Z coordinates.\n\n        Internal coordinates are bond length, angle and dihedral angles.\n\n        :param verbose bool: default False\n            describe runtime problems\n\n        '
        for chn in self.get_chains():
            chn.atom_to_internal_coordinates(verbose)

    def internal_to_atom_coordinates(self, verbose: bool=False) -> None:
        if False:
            while True:
                i = 10
        'Create/update atom coordinates from internal coordinates.\n\n        :param verbose bool: default False\n            describe runtime problems\n\n        :raises Exception: if any chain does not have .internal_coord attribute\n        '
        for chn in self.get_chains():
            chn.internal_to_atom_coordinates(verbose)