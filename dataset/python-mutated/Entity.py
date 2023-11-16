"""Base class for Residue, Chain, Model and Structure classes.

It is a simple container class, with list and dictionary like properties.
"""
from collections import deque
from copy import copy
import numpy as np
from Bio.PDB.PDBExceptions import PDBConstructionException

class Entity:
    """Basic container object for PDB hierarchy.

    Structure, Model, Chain and Residue are subclasses of Entity.
    It deals with storage and lookup.
    """

    def __init__(self, id):
        if False:
            return 10
        'Initialize the class.'
        self._id = id
        self.full_id = None
        self.parent = None
        self.child_list = []
        self.child_dict = {}
        self.xtra = {}

    def __len__(self):
        if False:
            print('Hello World!')
        'Return the number of children.'
        return len(self.child_list)

    def __getitem__(self, id):
        if False:
            print('Hello World!')
        'Return the child with given id.'
        return self.child_dict[id]

    def __delitem__(self, id):
        if False:
            return 10
        'Remove a child.'
        return self.detach_child(id)

    def __contains__(self, id):
        if False:
            while True:
                i = 10
        'Check if there is a child element with the given id.'
        return id in self.child_dict

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        'Iterate over children.'
        yield from self.child_list

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        'Test for equality. This compares full_id including the IDs of all parents.'
        if isinstance(other, type(self)):
            if self.parent is None:
                return self.id == other.id
            else:
                return self.full_id[1:] == other.full_id[1:]
        else:
            return NotImplemented

    def __ne__(self, other):
        if False:
            while True:
                i = 10
        'Test for inequality.'
        if isinstance(other, type(self)):
            if self.parent is None:
                return self.id != other.id
            else:
                return self.full_id[1:] != other.full_id[1:]
        else:
            return NotImplemented

    def __gt__(self, other):
        if False:
            return 10
        'Test greater than.'
        if isinstance(other, type(self)):
            if self.parent is None:
                return self.id > other.id
            else:
                return self.full_id[1:] > other.full_id[1:]
        else:
            return NotImplemented

    def __ge__(self, other):
        if False:
            return 10
        'Test greater or equal.'
        if isinstance(other, type(self)):
            if self.parent is None:
                return self.id >= other.id
            else:
                return self.full_id[1:] >= other.full_id[1:]
        else:
            return NotImplemented

    def __lt__(self, other):
        if False:
            i = 10
            return i + 15
        'Test less than.'
        if isinstance(other, type(self)):
            if self.parent is None:
                return self.id < other.id
            else:
                return self.full_id[1:] < other.full_id[1:]
        else:
            return NotImplemented

    def __le__(self, other):
        if False:
            while True:
                i = 10
        'Test less or equal.'
        if isinstance(other, type(self)):
            if self.parent is None:
                return self.id <= other.id
            else:
                return self.full_id[1:] <= other.full_id[1:]
        else:
            return NotImplemented

    def __hash__(self):
        if False:
            print('Hello World!')
        'Hash method to allow uniqueness (set).'
        return hash(self.full_id)

    def _reset_full_id(self):
        if False:
            print('Hello World!')
        'Reset the full_id (PRIVATE).\n\n        Resets the full_id of this entity and\n        recursively of all its children based on their ID.\n        '
        for child in self:
            try:
                child._reset_full_id()
            except AttributeError:
                pass
        self.full_id = self._generate_full_id()

    def _generate_full_id(self):
        if False:
            while True:
                i = 10
        'Generate full_id (PRIVATE).\n\n        Generate the full_id of the Entity based on its\n        Id and the IDs of the parents.\n        '
        entity_id = self.get_id()
        parts = [entity_id]
        parent = self.get_parent()
        while parent is not None:
            entity_id = parent.get_id()
            parts.append(entity_id)
            parent = parent.get_parent()
        parts.reverse()
        return tuple(parts)

    @property
    def id(self):
        if False:
            return 10
        'Return identifier.'
        return self._id

    @id.setter
    def id(self, value):
        if False:
            print('Hello World!')
        "Change the id of this entity.\n\n        This will update the child_dict of this entity's parent\n        and invalidate all cached full ids involving this entity.\n\n        @raises: ValueError\n        "
        if value == self._id:
            return
        if self.parent:
            if value in self.parent.child_dict:
                raise ValueError(f'Cannot change id from `{self._id}` to `{value}`. The id `{value}` is already used for a sibling of this entity.')
            del self.parent.child_dict[self._id]
            self.parent.child_dict[value] = self
        self._id = value
        self._reset_full_id()

    def get_level(self):
        if False:
            while True:
                i = 10
        'Return level in hierarchy.\n\n        A - atom\n        R - residue\n        C - chain\n        M - model\n        S - structure\n        '
        return self.level

    def set_parent(self, entity):
        if False:
            i = 10
            return i + 15
        'Set the parent Entity object.'
        self.parent = entity
        self._reset_full_id()

    def detach_parent(self):
        if False:
            return 10
        'Detach the parent.'
        self.parent = None

    def detach_child(self, id):
        if False:
            print('Hello World!')
        'Remove a child.'
        child = self.child_dict[id]
        child.detach_parent()
        del self.child_dict[id]
        self.child_list.remove(child)

    def add(self, entity):
        if False:
            print('Hello World!')
        'Add a child to the Entity.'
        entity_id = entity.get_id()
        if self.has_id(entity_id):
            raise PDBConstructionException(f'{entity_id} defined twice')
        entity.set_parent(self)
        self.child_list.append(entity)
        self.child_dict[entity_id] = entity

    def insert(self, pos, entity):
        if False:
            for i in range(10):
                print('nop')
        'Add a child to the Entity at a specified position.'
        entity_id = entity.get_id()
        if self.has_id(entity_id):
            raise PDBConstructionException(f'{entity_id} defined twice')
        entity.set_parent(self)
        self.child_list[pos:pos] = [entity]
        self.child_dict[entity_id] = entity

    def get_iterator(self):
        if False:
            print('Hello World!')
        'Return iterator over children.'
        yield from self.child_list

    def get_list(self):
        if False:
            return 10
        'Return a copy of the list of children.'
        return copy(self.child_list)

    def has_id(self, id):
        if False:
            return 10
        'Check if a child with given id exists.'
        return id in self.child_dict

    def get_parent(self):
        if False:
            return 10
        'Return the parent Entity object.'
        return self.parent

    def get_id(self):
        if False:
            while True:
                i = 10
        'Return the id.'
        return self.id

    def get_full_id(self):
        if False:
            i = 10
            return i + 15
        'Return the full id.\n\n        The full id is a tuple containing all id\'s starting from\n        the top object (Structure) down to the current object. A full id for\n        a Residue object e.g. is something like:\n\n        ("1abc", 0, "A", (" ", 10, "A"))\n\n        This corresponds to:\n\n        Structure with id "1abc"\n        Model with id 0\n        Chain with id "A"\n        Residue with id (" ", 10, "A")\n\n        The Residue id indicates that the residue is not a hetero-residue\n        (or a water) because it has a blank hetero field, that its sequence\n        identifier is 10 and its insertion code "A".\n        '
        if self.full_id is None:
            self.full_id = self._generate_full_id()
        return self.full_id

    def transform(self, rot, tran):
        if False:
            print('Hello World!')
        "Apply rotation and translation to the atomic coordinates.\n\n        :param rot: A right multiplying rotation matrix\n        :type rot: 3x3 NumPy array\n\n        :param tran: the translation vector\n        :type tran: size 3 NumPy array\n\n        Examples\n        --------\n        This is an incomplete but illustrative example::\n\n            from numpy import pi, array\n            from Bio.PDB.vectors import Vector, rotmat\n            rotation = rotmat(pi, Vector(1, 0, 0))\n            translation = array((0, 0, 1), 'f')\n            entity.transform(rotation, translation)\n\n        "
        for o in self.get_list():
            o.transform(rot, tran)

    def center_of_mass(self, geometric=False):
        if False:
            for i in range(10):
                print('nop')
        'Return the center of mass of the Entity as a numpy array.\n\n        If geometric is True, returns the center of geometry instead.\n        '
        if not len(self):
            raise ValueError(f'{self} does not have children')
        maybe_disordered = {'R', 'C'}
        only_atom_level = {'A'}
        entities = deque([self])
        while True:
            e = entities.popleft()
            if e.level in maybe_disordered:
                entities += e.get_unpacked_list()
            else:
                entities += e.child_list
            elevels = {e.level for e in entities}
            if elevels == only_atom_level:
                break
        coords = np.asarray([a.coord for a in entities], dtype=np.float32)
        if geometric:
            masses = None
        else:
            masses = np.asarray([a.mass for a in entities], dtype=np.float32)
        return np.average(coords, axis=0, weights=masses)

    def copy(self):
        if False:
            return 10
        'Copy entity recursively.'
        shallow = copy(self)
        shallow.child_list = []
        shallow.child_dict = {}
        shallow.xtra = copy(self.xtra)
        shallow.detach_parent()
        for child in self.child_list:
            shallow.add(child.copy())
        return shallow

class DisorderedEntityWrapper:
    """Wrapper class to group equivalent Entities.

    This class is a simple wrapper class that groups a number of equivalent
    Entities and forwards all method calls to one of them (the currently selected
    object). DisorderedResidue and DisorderedAtom are subclasses of this class.

    E.g.: A DisorderedAtom object contains a number of Atom objects,
    where each Atom object represents a specific position of a disordered
    atom in the structure.
    """

    def __init__(self, id):
        if False:
            while True:
                i = 10
        'Initialize the class.'
        self.id = id
        self.child_dict = {}
        self.selected_child = None
        self.parent = None

    def __getattr__(self, method):
        if False:
            print('Hello World!')
        'Forward the method call to the selected child.'
        if method == '__setstate__':
            raise AttributeError
        if not hasattr(self, 'selected_child'):
            raise AttributeError
        return getattr(self.selected_child, method)

    def __getitem__(self, id):
        if False:
            i = 10
            return i + 15
        'Return the child with the given id.'
        return self.selected_child[id]

    def __setitem__(self, id, child):
        if False:
            i = 10
            return i + 15
        'Add a child, associated with a certain id.'
        self.child_dict[id] = child

    def __contains__(self, id):
        if False:
            while True:
                i = 10
        'Check if the child has the given id.'
        return id in self.selected_child

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the number of children.'
        return iter(self.selected_child)

    def __len__(self):
        if False:
            print('Hello World!')
        'Return the number of children.'
        return len(self.selected_child)

    def __sub__(self, other):
        if False:
            print('Hello World!')
        'Subtraction with another object.'
        return self.selected_child - other

    def __gt__(self, other):
        if False:
            for i in range(10):
                print('nop')
        'Return if child is greater than other.'
        return self.selected_child > other

    def __ge__(self, other):
        if False:
            i = 10
            return i + 15
        'Return if child is greater or equal than other.'
        return self.selected_child >= other

    def __lt__(self, other):
        if False:
            i = 10
            return i + 15
        'Return if child is less than other.'
        return self.selected_child < other

    def __le__(self, other):
        if False:
            i = 10
            return i + 15
        'Return if child is less or equal than other.'
        return self.selected_child <= other

    def copy(self):
        if False:
            while True:
                i = 10
        'Copy disorderd entity recursively.'
        shallow = copy(self)
        shallow.child_dict = {}
        shallow.detach_parent()
        for child in self.disordered_get_list():
            shallow.disordered_add(child.copy())
        return shallow

    def get_id(self):
        if False:
            print('Hello World!')
        'Return the id.'
        return self.id

    def disordered_has_id(self, id):
        if False:
            while True:
                i = 10
        'Check if there is an object present associated with this id.'
        return id in self.child_dict

    def detach_parent(self):
        if False:
            for i in range(10):
                print('nop')
        'Detach the parent.'
        self.parent = None
        for child in self.disordered_get_list():
            child.detach_parent()

    def get_parent(self):
        if False:
            for i in range(10):
                print('nop')
        'Return parent.'
        return self.parent

    def set_parent(self, parent):
        if False:
            while True:
                i = 10
        'Set the parent for the object and its children.'
        self.parent = parent
        for child in self.disordered_get_list():
            child.set_parent(parent)

    def disordered_select(self, id):
        if False:
            print('Hello World!')
        'Select the object with given id as the currently active object.\n\n        Uncaught method calls are forwarded to the selected child object.\n        '
        self.selected_child = self.child_dict[id]

    def disordered_add(self, child):
        if False:
            for i in range(10):
                print('nop')
        'Add disordered entry.\n\n        This is implemented by DisorderedAtom and DisorderedResidue.\n        '
        raise NotImplementedError

    def disordered_remove(self, child):
        if False:
            i = 10
            return i + 15
        'Remove disordered entry.\n\n        This is implemented by DisorderedAtom and DisorderedResidue.\n        '
        raise NotImplementedError

    def is_disordered(self):
        if False:
            i = 10
            return i + 15
        'Return 2, indicating that this Entity is a collection of Entities.'
        return 2

    def disordered_get_id_list(self):
        if False:
            print('Hello World!')
        "Return a list of id's."
        return sorted(self.child_dict)

    def disordered_get(self, id=None):
        if False:
            while True:
                i = 10
        'Get the child object associated with id.\n\n        If id is None, the currently selected child is returned.\n        '
        if id is None:
            return self.selected_child
        return self.child_dict[id]

    def disordered_get_list(self):
        if False:
            for i in range(10):
                print('nop')
        'Return list of children.'
        return list(self.child_dict.values())