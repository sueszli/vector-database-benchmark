"""Chain class, used in Structure objects."""
from Bio.PDB.Entity import Entity
from Bio.PDB.internal_coords import IC_Chain
from typing import Optional

class Chain(Entity):
    """Define Chain class.

    Chain is an object of type Entity, stores residues and includes a method to
    access atoms from residues.
    """

    def __init__(self, id):
        if False:
            while True:
                i = 10
        'Initialize the class.'
        self.level = 'C'
        self.internal_coord = None
        Entity.__init__(self, id)

    def __gt__(self, other):
        if False:
            i = 10
            return i + 15
        'Validate if id is greater than other.id.'
        if isinstance(other, Chain):
            if self.id == ' ' and other.id != ' ':
                return 0
            elif self.id != ' ' and other.id == ' ':
                return 1
            else:
                return self.id > other.id
        else:
            return NotImplemented

    def __ge__(self, other):
        if False:
            return 10
        'Validate if id is greater or equal than other.id.'
        if isinstance(other, Chain):
            if self.id == ' ' and other.id != ' ':
                return 0
            elif self.id != ' ' and other.id == ' ':
                return 1
            else:
                return self.id >= other.id
        else:
            return NotImplemented

    def __lt__(self, other):
        if False:
            print('Hello World!')
        'Validate if id is less than other.id.'
        if isinstance(other, Chain):
            if self.id == ' ' and other.id != ' ':
                return 0
            elif self.id != ' ' and other.id == ' ':
                return 1
            else:
                return self.id < other.id
        else:
            return NotImplemented

    def __le__(self, other):
        if False:
            while True:
                i = 10
        'Validate if id is less or equal than other id.'
        if isinstance(other, Chain):
            if self.id == ' ' and other.id != ' ':
                return 0
            elif self.id != ' ' and other.id == ' ':
                return 1
            else:
                return self.id <= other.id
        else:
            return NotImplemented

    def _translate_id(self, id):
        if False:
            for i in range(10):
                print('nop')
        'Translate sequence identifier to tuple form (PRIVATE).\n\n        A residue id is normally a tuple (hetero flag, sequence identifier,\n        insertion code). Since for most residues the hetero flag and the\n        insertion code are blank (i.e. " "), you can just use the sequence\n        identifier to index a residue in a chain. The _translate_id method\n        translates the sequence identifier to the (" ", sequence identifier,\n        " ") tuple.\n\n        Arguments:\n         - id - int, residue resseq\n\n        '
        if isinstance(id, int):
            id = (' ', id, ' ')
        return id

    def __getitem__(self, id):
        if False:
            return 10
        'Return the residue with given id.\n\n        The id of a residue is (hetero flag, sequence identifier, insertion code).\n        If id is an int, it is translated to (" ", id, " ") by the _translate_id\n        method.\n\n        Arguments:\n         - id - (string, int, string) or int\n\n        '
        id = self._translate_id(id)
        return Entity.__getitem__(self, id)

    def __contains__(self, id):
        if False:
            while True:
                i = 10
        'Check if a residue with given id is present in this chain.\n\n        Arguments:\n         - id - (string, int, string) or int\n\n        '
        id = self._translate_id(id)
        return Entity.__contains__(self, id)

    def __delitem__(self, id):
        if False:
            for i in range(10):
                print('nop')
        'Delete item.\n\n        Arguments:\n         - id - (string, int, string) or int\n\n        '
        id = self._translate_id(id)
        return Entity.__delitem__(self, id)

    def __repr__(self):
        if False:
            while True:
                i = 10
        'Return the chain identifier.'
        return f'<Chain id={self.get_id()}>'

    def get_unpacked_list(self):
        if False:
            i = 10
            return i + 15
        'Return a list of undisordered residues.\n\n        Some Residue objects hide several disordered residues\n        (DisorderedResidue objects). This method unpacks them,\n        ie. it returns a list of simple Residue objects.\n        '
        unpacked_list = []
        for residue in self.get_list():
            if residue.is_disordered() == 2:
                for dresidue in residue.disordered_get_list():
                    unpacked_list.append(dresidue)
            else:
                unpacked_list.append(residue)
        return unpacked_list

    def has_id(self, id):
        if False:
            while True:
                i = 10
        'Return 1 if a residue with given id is present.\n\n        The id of a residue is (hetero flag, sequence identifier, insertion code).\n\n        If id is an int, it is translated to (" ", id, " ") by the _translate_id\n        method.\n\n        Arguments:\n         - id - (string, int, string) or int\n\n        '
        id = self._translate_id(id)
        return Entity.has_id(self, id)

    def get_residues(self):
        if False:
            print('Hello World!')
        'Return residues.'
        yield from self

    def get_atoms(self):
        if False:
            return 10
        'Return atoms from residues.'
        for r in self.get_residues():
            yield from r

    def atom_to_internal_coordinates(self, verbose: bool=False) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Create/update internal coordinates from Atom X,Y,Z coordinates.\n\n        Internal coordinates are bond length, angle and dihedral angles.\n\n        :param verbose bool: default False\n            describe runtime problems\n        '
        if not self.internal_coord:
            self.internal_coord = IC_Chain(self, verbose)
        self.internal_coord.atom_to_internal_coordinates(verbose=verbose)

    def internal_to_atom_coordinates(self, verbose: bool=False, start: Optional[int]=None, fin: Optional[int]=None):
        if False:
            print('Hello World!')
        'Create/update atom coordinates from internal coordinates.\n\n        :param verbose bool: default False\n            describe runtime problems\n        :param: start, fin integers\n            optional sequence positions for begin, end of subregion to process.\n            N.B. this activates serial residue assembly, <start> residue CA will\n            be at origin\n        :raises Exception: if any chain does not have .internal_coord attribute\n        '
        if self.internal_coord:
            self.internal_coord.internal_to_atom_coordinates(verbose=verbose, start=start, fin=fin)
        else:
            raise Exception('Structure %s Chain %s does not have internal coordinates set' % (self.parent.parent, self))