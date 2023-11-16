""" Helper for portable metaclasses that do checks. """
from abc import ABCMeta
from nuitka.Errors import NuitkaNodeDesignError

def getMetaClassBase(meta_class_prefix, require_slots):
    if False:
        while True:
            i = 10
    'For Python2/3 compatible source, we create a base class that has the metaclass\n    used and doesn\'t require making a syntax choice.\n\n    Also this allows to enforce the proper usage of "__slots__" for all classes using\n    it optionally.\n    '

    class MetaClass(ABCMeta):

        def __new__(mcs, name, bases, dictionary):
            if False:
                i = 10
                return i + 15
            if require_slots:
                for base in bases:
                    if base is not object and '__slots__' not in base.__dict__:
                        raise NuitkaNodeDesignError(name, 'All bases must set __slots__.', base)
                if '__slots__' not in dictionary:
                    raise NuitkaNodeDesignError(name, 'Class must set __slots__.', name)
            return ABCMeta.__new__(mcs, name, bases, dictionary)
    MetaClassBase = MetaClass('%sMetaClassBase' % meta_class_prefix, (object,), {'__slots__': ()} if require_slots else {})
    return MetaClassBase