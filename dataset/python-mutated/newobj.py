"""
@author:       Brendan Dolan-Gavitt
@license:      GNU General Public License 2.0 or later
@contact:      bdolangavitt@wesleyan.edu
"""
from object import *
from types import regtypes as types
from operator import itemgetter
from struct import unpack

def get_ptr_type(structure, member):
    if False:
        print('Hello World!')
    "Return the type a pointer points to.\n       \n       Arguments:\n         structure : the name of the structure from vtypes\n         member : a list of members\n\n       Example:\n         get_ptr_type('_EPROCESS', ['ActiveProcessLinks', 'Flink']) => ['_LIST_ENTRY']\n    "
    if len(member) > 1:
        (_, tp) = get_obj_offset(types, [structure, member[0]])
        if tp == 'array':
            return types[structure][1][member[0]][1][2][1]
        else:
            return get_ptr_type(tp, member[1:])
    else:
        return types[structure][1][member[0]][1][1]

class Obj(object):
    """Base class for all objects.
       
       May return a subclass for certain data types to allow
       for special handling.
    """

    def __new__(typ, name, address, space):
        if False:
            print('Hello World!')
        if name in globals():
            return globals()[name](name, address, space)
        elif name in builtin_types:
            return Primitive(name, address, space)
        else:
            obj = object.__new__(typ)
            return obj

    def __init__(self, name, address, space):
        if False:
            return 10
        self.name = name
        self.address = address
        self.space = space
        self.extra_members = []

    def __getattribute__(self, attr):
        if False:
            i = 10
            return i + 15
        try:
            return object.__getattribute__(self, attr)
        except AttributeError:
            pass
        if self.name in builtin_types:
            raise AttributeError('Primitive types have no dynamic attributes')
        try:
            (off, tp) = get_obj_offset(types, [self.name, attr])
        except:
            raise AttributeError("'%s' has no attribute '%s'" % (self.name, attr))
        if tp == 'array':
            a_len = types[self.name][1][attr][1][1]
            l = []
            for i in range(a_len):
                (a_off, a_tp) = get_obj_offset(types, [self.name, attr, i])
                if a_tp == 'pointer':
                    ptp = get_ptr_type(self.name, [attr, i])
                    l.append(Pointer(a_tp, self.address + a_off, self.space, ptp))
                else:
                    l.append(Obj(a_tp, self.address + a_off, self.space))
            return l
        elif tp == 'pointer':
            ptp = get_ptr_type(self.name, [attr])
            return Pointer(tp, self.address + off, self.space, ptp)
        else:
            return Obj(tp, self.address + off, self.space)

    def __div__(self, other):
        if False:
            while True:
                i = 10
        if isinstance(other, tuple) or isinstance(other, list):
            return Pointer(other[0], self.address, self.space, other[1])
        elif isinstance(other, str):
            return Obj(other, self.address, self.space)
        else:
            raise ValueError('Must provide a type name as string for casting')

    def members(self):
        if False:
            for i in range(10):
                print('nop')
        "Return a list of this object's members, sorted by offset."
        membs = [(k, v[0]) for (k, v) in types[self.name][1].items()]
        membs.sort(key=itemgetter(1))
        return map(itemgetter(0), membs) + self.extra_members

    def values(self):
        if False:
            while True:
                i = 10
        "Return a dictionary of this object's members and their values"
        valdict = {}
        for k in self.members():
            valdict[k] = getattr(self, k)
        return valdict

    def bytes(self, length=-1):
        if False:
            print('Hello World!')
        'Get bytes starting at the address of this object.\n        \n           Arguments:\n             length : the number of bytes to read. Default: size of\n                this object.\n        '
        if length == -1:
            length = self.size()
        return self.space.read(self.address, length)

    def size(self):
        if False:
            while True:
                i = 10
        'Get the size of this object.'
        if self.name in builtin_types:
            return builtin_types[self.name][0]
        else:
            return types[self.name][0]

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return '<%s @%08x>' % (self.name, self.address)

    def __eq__(self, other):
        if False:
            print('Hello World!')
        if not isinstance(other, Obj):
            raise TypeError('Types are incomparable')
        return self.address == other.address and self.name == other.name

    def __ne__(self, other):
        if False:
            print('Hello World!')
        return not self.__eq__(other)

    def __hash__(self):
        if False:
            print('Hello World!')
        return hash(self.address) ^ hash(self.name)

    def is_valid(self):
        if False:
            return 10
        return self.space.is_valid_address(self.address)

    def get_offset(self, member):
        if False:
            for i in range(10):
                print('nop')
        return get_obj_offset(types, [self.name] + member)

class Primitive(Obj):
    """Class to represent a primitive data type.
       
       Attributes:
         value : the python primitive value of this type
    """

    def __new__(typ, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        obj = object.__new__(typ)
        return obj

    def __init__(self, name, address, space):
        if False:
            for i in range(10):
                print('nop')
        super(Primitive, self).__init__(name, address, space)
        (length, fmt) = builtin_types[name]
        data = space.read(address, length)
        if not data:
            self.value = None
        else:
            self.value = unpack(fmt, data)[0]

    def __repr__(self):
        if False:
            return 10
        return repr(self.value)

    def members(self):
        if False:
            while True:
                i = 10
        return []

class Pointer(Obj):
    """Class to represent pointers.
    
       value : the object pointed to

       If an attribute is not found in this instance,
       the attribute will be looked up in the referenced
       object."""

    def __new__(typ, *args, **kwargs):
        if False:
            while True:
                i = 10
        obj = object.__new__(typ)
        return obj

    def __init__(self, name, address, space, ptr_type):
        if False:
            while True:
                i = 10
        super(Pointer, self).__init__(name, address, space)
        ptr_address = read_value(space, name, address)
        if ptr_type[0] == 'pointer':
            self.value = Pointer(ptr_type[0], ptr_address, self.space, ptr_type[1])
        else:
            self.value = Obj(ptr_type[0], ptr_address, self.space)

    def __getattribute__(self, attr):
        if False:
            print('Hello World!')
        try:
            return super(Pointer, self).__getattribute__(attr)
        except AttributeError:
            return getattr(self.value, attr)

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return '<pointer to [%s @%08x]>' % (self.value.name, self.value.address)

    def members(self):
        if False:
            i = 10
            return i + 15
        return self.value.members()

class _UNICODE_STRING(Obj):
    """Class representing a _UNICODE_STRING

    Adds the following behavior:
      * The Buffer attribute is presented as a Python string rather
        than a pointer to an unsigned short.
      * The __str__ method returns the value of the Buffer.
    """

    def __new__(typ, *args, **kwargs):
        if False:
            while True:
                i = 10
        obj = object.__new__(typ)
        return obj

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return self.Buffer

    def getBuffer(self):
        if False:
            i = 10
            return i + 15
        return read_unicode_string(self.space, types, [], self.address)
    Buffer = property(fget=getBuffer)

class _CM_KEY_NODE(Obj):

    def __new__(typ, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        obj = object.__new__(typ)
        return obj

    def getName(self):
        if False:
            while True:
                i = 10
        return read_string(self.space, types, ['_CM_KEY_NODE', 'Name'], self.address, self.NameLength.value)
    Name = property(fget=getName)

class _CM_KEY_VALUE(Obj):

    def __new__(typ, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        obj = object.__new__(typ)
        return obj

    def getName(self):
        if False:
            i = 10
            return i + 15
        return read_string(self.space, types, ['_CM_KEY_VALUE', 'Name'], self.address, self.NameLength.value)
    Name = property(fget=getName)

class _CHILD_LIST(Obj):

    def __new__(typ, *args, **kwargs):
        if False:
            print('Hello World!')
        obj = object.__new__(typ)
        return obj

    def getList(self):
        if False:
            i = 10
            return i + 15
        lst = []
        list_address = read_obj(self.space, types, ['_CHILD_LIST', 'List'], self.address)
        for i in range(self.Count.value):
            lst.append(Pointer('pointer', list_address + i * 4, self.space, ['_CM_KEY_VALUE']))
        return lst
    List = property(fget=getList)

class _CM_KEY_INDEX(Obj):

    def __new__(typ, *args, **kwargs):
        if False:
            return 10
        obj = object.__new__(typ)
        return obj

    def getList(self):
        if False:
            print('Hello World!')
        lst = []
        for i in range(self.Count.value):
            (off, tp) = get_obj_offset(types, ['_CM_KEY_INDEX', 'List', i * 2])
            lst.append(Pointer('pointer', self.address + off, self.space, ['_CM_KEY_NODE']))
        return lst
    List = property(fget=getList)