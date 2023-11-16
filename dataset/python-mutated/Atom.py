"""Atom class, used in Structure objects."""
import copy
import sys
import warnings
import numpy as np
from Bio.PDB.Entity import DisorderedEntityWrapper
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio.PDB.vectors import Vector
from Bio.Data import IUPACData

class Atom:
    """Define Atom class.

    The Atom object stores atom name (both with and without spaces),
    coordinates, B factor, occupancy, alternative location specifier
    and (optionally) anisotropic B factor and standard deviations of
    B factor and positions.

    In the case of PQR files, B factor and occupancy are replaced by
    atomic charge and radius.
    """

    def __init__(self, name, coord, bfactor, occupancy, altloc, fullname, serial_number, element=None, pqr_charge=None, radius=None):
        if False:
            print('Hello World!')
        'Initialize Atom object.\n\n        :param name: atom name (eg. "CA"). Note that spaces are normally stripped.\n        :type name: string\n\n        :param coord: atomic coordinates (x,y,z)\n        :type coord: NumPy array (Float0, length 3)\n\n        :param bfactor: isotropic B factor\n        :type bfactor: number\n\n        :param occupancy: occupancy (0.0-1.0)\n        :type occupancy: number\n\n        :param altloc: alternative location specifier for disordered atoms\n        :type altloc: string\n\n        :param fullname: full atom name, including spaces, e.g. " CA ". Normally\n                         these spaces are stripped from the atom name.\n        :type fullname: string\n\n        :param element: atom element, e.g. "C" for Carbon, "HG" for mercury,\n        :type element: uppercase string (or None if unknown)\n\n        :param pqr_charge: atom charge\n        :type pqr_charge: number\n\n        :param radius: atom radius\n        :type radius: number\n        '
        self.level = 'A'
        self.parent = None
        self.name = name
        self.fullname = fullname
        self.coord = coord
        self.bfactor = bfactor
        self.occupancy = occupancy
        self.altloc = altloc
        self.full_id = None
        self.id = name
        self.disordered_flag = 0
        self.anisou_array = None
        self.siguij_array = None
        self.sigatm_array = None
        self.serial_number = serial_number
        self.xtra = {}
        assert not element or element == element.upper(), element
        self.element = self._assign_element(element)
        self.mass = self._assign_atom_mass()
        self.pqr_charge = pqr_charge
        self.radius = radius
        self._sorting_keys = {'N': 0, 'CA': 1, 'C': 2, 'O': 3}

    def __eq__(self, other):
        if False:
            while True:
                i = 10
        'Test equality.'
        if isinstance(other, Atom):
            return self.full_id[1:] == other.full_id[1:]
        else:
            return NotImplemented

    def __ne__(self, other):
        if False:
            print('Hello World!')
        'Test inequality.'
        if isinstance(other, Atom):
            return self.full_id[1:] != other.full_id[1:]
        else:
            return NotImplemented

    def __gt__(self, other):
        if False:
            print('Hello World!')
        'Test greater than.'
        if isinstance(other, Atom):
            if self.parent != other.parent:
                return self.parent > other.parent
            order_s = self._sorting_keys.get(self.name, 4)
            order_o = self._sorting_keys.get(other.name, 4)
            if order_s != order_o:
                return order_s > order_o
            elif self.name != other.name:
                return self.name > other.name
            else:
                return self.altloc > other.altloc
        else:
            return NotImplemented

    def __ge__(self, other):
        if False:
            for i in range(10):
                print('nop')
        'Test greater or equal.'
        if isinstance(other, Atom):
            if self.parent != other.parent:
                return self.parent >= other.parent
            order_s = self._sorting_keys.get(self.name, 4)
            order_o = self._sorting_keys.get(other.name, 4)
            if order_s != order_o:
                return order_s >= order_o
            elif self.name != other.name:
                return self.name >= other.name
            else:
                return self.altloc >= other.altloc
        else:
            return NotImplemented

    def __lt__(self, other):
        if False:
            for i in range(10):
                print('nop')
        'Test less than.'
        if isinstance(other, Atom):
            if self.parent != other.parent:
                return self.parent < other.parent
            order_s = self._sorting_keys.get(self.name, 4)
            order_o = self._sorting_keys.get(other.name, 4)
            if order_s != order_o:
                return order_s < order_o
            elif self.name != other.name:
                return self.name < other.name
            else:
                return self.altloc < other.altloc
        else:
            return NotImplemented

    def __le__(self, other):
        if False:
            for i in range(10):
                print('nop')
        'Test less or equal.'
        if isinstance(other, Atom):
            if self.parent != other.parent:
                return self.parent <= other.parent
            order_s = self._sorting_keys.get(self.name, 4)
            order_o = self._sorting_keys.get(other.name, 4)
            if order_s != order_o:
                return order_s <= order_o
            elif self.name != other.name:
                return self.name <= other.name
            else:
                return self.altloc <= other.altloc
        else:
            return NotImplemented

    def __hash__(self):
        if False:
            return 10
        'Return atom full identifier.'
        return hash(self.get_full_id())

    def _assign_element(self, element):
        if False:
            return 10
        'Guess element from atom name if not recognised (PRIVATE).\n\n        There is little documentation about extracting/encoding element\n        information in atom names, but some conventions seem to prevail:\n\n            - C, N, O, S, H, P, F atom names start with a blank space (e.g. " CA ")\n              unless the name is 4 characters long (e.g. HE21 in glutamine). In both\n              these cases, the element is the first character.\n\n            - Inorganic elements do not have a blank space (e.g. "CA  " for calcium)\n              but one must check the full name to differentiate between e.g. helium\n              ("HE  ") and long-name hydrogens (e.g. "HE21").\n\n            - Atoms with unknown or ambiguous elements are marked with \'X\', e.g.\n              PDB 4cpa. If we fail to identify an element, we should mark it as\n              such.\n\n        '
        if not element or element.capitalize() not in IUPACData.atom_weights:
            if self.fullname[0].isalpha() and (not self.fullname[2:].isdigit()):
                putative_element = self.name.strip()
            elif self.name[0].isdigit():
                putative_element = self.name[1]
            else:
                putative_element = self.name[0]
            if putative_element.capitalize() in IUPACData.atom_weights:
                msg = 'Used element %r for Atom (name=%s) with given element %r' % (putative_element, self.name, element)
                element = putative_element
            else:
                msg = 'Could not assign element %r for Atom (name=%s) with given element %r' % (putative_element, self.name, element)
                element = 'X'
            warnings.warn(msg, PDBConstructionWarning)
        return element

    def _assign_atom_mass(self):
        if False:
            while True:
                i = 10
        'Return atom weight (PRIVATE).'
        try:
            return IUPACData.atom_weights[self.element.capitalize()]
        except (AttributeError, KeyError):
            return float('NaN')

    def __repr__(self):
        if False:
            return 10
        'Print Atom object as <Atom atom_name>.'
        return f'<Atom {self.get_id()}>'

    def __sub__(self, other):
        if False:
            i = 10
            return i + 15
        'Calculate distance between two atoms.\n\n        :param other: the other atom\n        :type other: L{Atom}\n\n        Examples\n        --------\n        This is an incomplete but illustrative example::\n\n            distance = atom1 - atom2\n\n        '
        diff = self.coord - other.coord
        return np.sqrt(np.dot(diff, diff))

    def set_serial_number(self, n):
        if False:
            print('Hello World!')
        'Set serial number.'
        self.serial_number = n

    def set_bfactor(self, bfactor):
        if False:
            print('Hello World!')
        'Set isotroptic B factor.'
        self.bfactor = bfactor

    def set_coord(self, coord):
        if False:
            print('Hello World!')
        'Set coordinates.'
        self.coord = coord

    def set_altloc(self, altloc):
        if False:
            for i in range(10):
                print('nop')
        'Set alternative location specifier.'
        self.altloc = altloc

    def set_occupancy(self, occupancy):
        if False:
            i = 10
            return i + 15
        'Set occupancy.'
        self.occupancy = occupancy

    def set_sigatm(self, sigatm_array):
        if False:
            return 10
        'Set standard deviation of atomic parameters.\n\n        The standard deviation of atomic parameters consists\n        of 3 positional, 1 B factor and 1 occupancy standard\n        deviation.\n\n        :param sigatm_array: standard deviations of atomic parameters.\n        :type sigatm_array: NumPy array (length 5)\n        '
        self.sigatm_array = sigatm_array

    def set_siguij(self, siguij_array):
        if False:
            return 10
        'Set standard deviations of anisotropic temperature factors.\n\n        :param siguij_array: standard deviations of anisotropic temperature factors.\n        :type siguij_array: NumPy array (length 6)\n        '
        self.siguij_array = siguij_array

    def set_anisou(self, anisou_array):
        if False:
            print('Hello World!')
        'Set anisotropic B factor.\n\n        :param anisou_array: anisotropic B factor.\n        :type anisou_array: NumPy array (length 6)\n        '
        self.anisou_array = anisou_array

    def set_charge(self, pqr_charge):
        if False:
            for i in range(10):
                print('nop')
        'Set charge.'
        self.pqr_charge = pqr_charge

    def set_radius(self, radius):
        if False:
            i = 10
            return i + 15
        'Set radius.'
        self.radius = radius

    def flag_disorder(self):
        if False:
            print('Hello World!')
        'Set the disordered flag to 1.\n\n        The disordered flag indicates whether the atom is disordered or not.\n        '
        self.disordered_flag = 1

    def is_disordered(self):
        if False:
            while True:
                i = 10
        'Return the disordered flag (1 if disordered, 0 otherwise).'
        return self.disordered_flag

    def set_parent(self, parent):
        if False:
            i = 10
            return i + 15
        'Set the parent residue.\n\n        Arguments:\n         - parent - Residue object\n\n        '
        self.parent = parent
        self.full_id = self.get_full_id()

    def detach_parent(self):
        if False:
            print('Hello World!')
        'Remove reference to parent.'
        self.parent = None

    def get_sigatm(self):
        if False:
            print('Hello World!')
        'Return standard deviation of atomic parameters.'
        return self.sigatm_array

    def get_siguij(self):
        if False:
            print('Hello World!')
        'Return standard deviations of anisotropic temperature factors.'
        return self.siguij_array

    def get_anisou(self):
        if False:
            i = 10
            return i + 15
        'Return anisotropic B factor.'
        return self.anisou_array

    def get_parent(self):
        if False:
            return 10
        'Return parent residue.'
        return self.parent

    def get_serial_number(self):
        if False:
            print('Hello World!')
        'Return the serial number.'
        return self.serial_number

    def get_name(self):
        if False:
            print('Hello World!')
        'Return atom name.'
        return self.name

    def get_id(self):
        if False:
            while True:
                i = 10
        'Return the id of the atom (which is its atom name).'
        return self.id

    def get_full_id(self):
        if False:
            print('Hello World!')
        'Return the full id of the atom.\n\n        The full id of an atom is a tuple used to uniquely identify\n        the atom and consists of the following elements:\n        (structure id, model id, chain id, residue id, atom name, altloc)\n        '
        try:
            return self.parent.get_full_id() + ((self.name, self.altloc),)
        except AttributeError:
            return (None, None, None, None, self.name, self.altloc)

    def get_coord(self):
        if False:
            print('Hello World!')
        'Return atomic coordinates.'
        return self.coord

    def get_bfactor(self):
        if False:
            i = 10
            return i + 15
        'Return B factor.'
        return self.bfactor

    def get_occupancy(self):
        if False:
            i = 10
            return i + 15
        'Return occupancy.'
        return self.occupancy

    def get_fullname(self):
        if False:
            i = 10
            return i + 15
        'Return the atom name, including leading and trailing spaces.'
        return self.fullname

    def get_altloc(self):
        if False:
            while True:
                i = 10
        'Return alternative location specifier.'
        return self.altloc

    def get_level(self):
        if False:
            for i in range(10):
                print('nop')
        'Return level.'
        return self.level

    def get_charge(self):
        if False:
            return 10
        'Return charge.'
        return self.pqr_charge

    def get_radius(self):
        if False:
            while True:
                i = 10
        'Return radius.'
        return self.radius

    def transform(self, rot, tran):
        if False:
            for i in range(10):
                print('nop')
        "Apply rotation and translation to the atomic coordinates.\n\n        :param rot: A right multiplying rotation matrix\n        :type rot: 3x3 NumPy array\n\n        :param tran: the translation vector\n        :type tran: size 3 NumPy array\n\n        Examples\n        --------\n        This is an incomplete but illustrative example::\n\n            from numpy import pi, array\n            from Bio.PDB.vectors import Vector, rotmat\n            rotation = rotmat(pi, Vector(1, 0, 0))\n            translation = array((0, 0, 1), 'f')\n            atom.transform(rotation, translation)\n\n        "
        self.coord = np.dot(self.coord, rot) + tran

    def get_vector(self):
        if False:
            i = 10
            return i + 15
        'Return coordinates as Vector.\n\n        :return: coordinates as 3D vector\n        :rtype: Bio.PDB.Vector class\n        '
        (x, y, z) = self.coord
        return Vector(x, y, z)

    def copy(self):
        if False:
            for i in range(10):
                print('nop')
        'Create a copy of the Atom.\n\n        Parent information is lost.\n        '
        shallow = copy.copy(self)
        shallow.detach_parent()
        shallow.set_coord(copy.copy(self.get_coord()))
        shallow.xtra = self.xtra.copy()
        return shallow

class DisorderedAtom(DisorderedEntityWrapper):
    """Contains all Atom objects that represent the same disordered atom.

    One of these atoms is "selected" and all method calls not caught
    by DisorderedAtom are forwarded to the selected Atom object. In that way, a
    DisorderedAtom behaves exactly like a normal Atom. By default, the selected
    Atom object represents the Atom object with the highest occupancy, but a
    different Atom object can be selected by using the disordered_select(altloc)
    method.
    """

    def __init__(self, id):
        if False:
            for i in range(10):
                print('nop')
        'Create DisorderedAtom.\n\n        Arguments:\n         - id - string, atom name\n\n        '
        self.last_occupancy = -sys.maxsize
        DisorderedEntityWrapper.__init__(self, id)

    def __iter__(self):
        if False:
            while True:
                i = 10
        'Iterate through disordered atoms.'
        yield from self.disordered_get_list()

    def __repr__(self):
        if False:
            print('Hello World!')
        'Return disordered atom identifier.'
        if self.child_dict:
            return f'<DisorderedAtom {self.get_id()}>'
        else:
            return f'<Empty DisorderedAtom {self.get_id()}>'

    def center_of_mass(self):
        if False:
            i = 10
            return i + 15
        'Return the center of mass of the DisorderedAtom as a numpy array.\n\n        Assumes all child atoms have the same mass (same element).\n        '
        children = self.disordered_get_list()
        if not children:
            raise ValueError(f'{self} does not have children')
        coords = np.asarray([a.coord for a in children], dtype=np.float32)
        return np.average(coords, axis=0, weights=None)

    def disordered_get_list(self):
        if False:
            return 10
        'Return list of atom instances.\n\n        Sorts children by altloc (empty, then alphabetical).\n        '
        return sorted(self.child_dict.values(), key=lambda a: ord(a.altloc))

    def disordered_add(self, atom):
        if False:
            return 10
        'Add a disordered atom.'
        atom.flag_disorder()
        residue = self.get_parent()
        atom.set_parent(residue)
        altloc = atom.get_altloc()
        occupancy = atom.get_occupancy()
        self[altloc] = atom
        if occupancy > self.last_occupancy:
            self.last_occupancy = occupancy
            self.disordered_select(altloc)

    def disordered_remove(self, altloc):
        if False:
            return 10
        'Remove a child atom altloc from the DisorderedAtom.\n\n        Arguments:\n         - altloc - name of the altloc to remove, as a string.\n\n        '
        atom = self.child_dict[altloc]
        is_selected = self.selected_child is atom
        del self.child_dict[altloc]
        atom.detach_parent()
        if is_selected and self.child_dict:
            child = sorted(self.child_dict.values(), key=lambda a: a.occupancy)[-1]
            self.disordered_select(child.altloc)
        elif not self.child_dict:
            self.selected_child = None
            self.last_occupancy = -sys.maxsize

    def transform(self, rot, tran):
        if False:
            for i in range(10):
                print('nop')
        'Apply rotation and translation to all children.\n\n        See the documentation of Atom.transform for details.\n        '
        for child in self:
            child.coord = np.dot(child.coord, rot) + tran