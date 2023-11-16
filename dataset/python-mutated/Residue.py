"""Residue class, used by Structure objects."""
from Bio.PDB.PDBExceptions import PDBConstructionException
from Bio.PDB.Entity import Entity, DisorderedEntityWrapper
_atom_name_dict = {}
_atom_name_dict['N'] = 1
_atom_name_dict['CA'] = 2
_atom_name_dict['C'] = 3
_atom_name_dict['O'] = 4

class Residue(Entity):
    """Represents a residue. A Residue object stores atoms."""

    def __init__(self, id, resname, segid):
        if False:
            i = 10
            return i + 15
        'Initialize the class.'
        self.level = 'R'
        self.disordered = 0
        self.resname = resname
        self.segid = segid
        self.internal_coord = None
        Entity.__init__(self, id)

    def __repr__(self):
        if False:
            return 10
        'Return the residue full id.'
        resname = self.get_resname()
        (hetflag, resseq, icode) = self.get_id()
        full_id = (resname, hetflag, resseq, icode)
        return '<Residue %s het=%s resseq=%s icode=%s>' % full_id

    def add(self, atom):
        if False:
            return 10
        'Add an Atom object.\n\n        Checks for adding duplicate atoms, and raises a\n        PDBConstructionException if so.\n        '
        atom_id = atom.get_id()
        if self.has_id(atom_id):
            raise PDBConstructionException(f'Atom {atom_id} defined twice in residue {self}')
        Entity.add(self, atom)

    def flag_disordered(self):
        if False:
            i = 10
            return i + 15
        'Set the disordered flag.'
        self.disordered = 1

    def is_disordered(self):
        if False:
            while True:
                i = 10
        'Return 1 if the residue contains disordered atoms.'
        return self.disordered

    def get_resname(self):
        if False:
            while True:
                i = 10
        'Return the residue name.'
        return self.resname

    def get_unpacked_list(self):
        if False:
            print('Hello World!')
        'Return the list of all atoms, unpack DisorderedAtoms.'
        atom_list = self.get_list()
        undisordered_atom_list = []
        for atom in atom_list:
            if atom.is_disordered():
                undisordered_atom_list += atom.disordered_get_list()
            else:
                undisordered_atom_list.append(atom)
        return undisordered_atom_list

    def get_segid(self):
        if False:
            while True:
                i = 10
        'Return the segment identifier.'
        return self.segid

    def get_atoms(self):
        if False:
            print('Hello World!')
        'Return atoms.'
        yield from self

class DisorderedResidue(DisorderedEntityWrapper):
    """DisorderedResidue is a wrapper around two or more Residue objects.

    It is used to represent point mutations (e.g. there is a Ser 60 and a Cys 60
    residue, each with 50 % occupancy).
    """

    def __init__(self, id):
        if False:
            while True:
                i = 10
        'Initialize the class.'
        DisorderedEntityWrapper.__init__(self, id)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        'Return disordered residue full identifier.'
        if self.child_dict:
            resname = self.get_resname()
            (hetflag, resseq, icode) = self.get_id()
            full_id = (resname, hetflag, resseq, icode)
            return '<DisorderedResidue %s het=%s resseq=%i icode=%s>' % full_id
        else:
            return '<Empty DisorderedResidue>'

    def add(self, atom):
        if False:
            for i in range(10):
                print('nop')
        'Add atom to residue.'
        residue = self.disordered_get()
        if atom.is_disordered() != 2:
            resname = residue.get_resname()
            (het, resseq, icode) = residue.get_id()
            residue.add(atom)
            raise PDBConstructionException('Blank altlocs in duplicate residue %s (%s, %i, %s)' % (resname, het, resseq, icode))
        residue.add(atom)

    def sort(self):
        if False:
            return 10
        'Sort the atoms in the child Residue objects.'
        for residue in self.disordered_get_list():
            residue.sort()

    def disordered_add(self, residue):
        if False:
            return 10
        'Add a residue object and use its resname as key.\n\n        Arguments:\n         - residue - Residue object\n\n        '
        resname = residue.get_resname()
        chain = self.get_parent()
        residue.set_parent(chain)
        assert not self.disordered_has_id(resname)
        self[resname] = residue
        self.disordered_select(resname)

    def disordered_remove(self, resname):
        if False:
            while True:
                i = 10
        'Remove a child residue from the DisorderedResidue.\n\n        Arguments:\n         - resname - name of the child residue to remove, as a string.\n\n        '
        residue = self.child_dict[resname]
        is_selected = self.selected_child is residue
        del self.child_dict[resname]
        residue.detach_parent()
        if is_selected and self.child_dict:
            child = next(iter(self.child_dict))
            self.disordered_select(child)
        elif not self.child_dict:
            self.selected_child = None