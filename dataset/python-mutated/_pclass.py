from pyrsistent._checked_types import InvariantException, CheckedType, _restore_pickle, store_invariants
from pyrsistent._field_common import set_fields, check_type, is_field_ignore_extra_complaint, PFIELD_NO_INITIAL, serialize, check_global_invariants
from pyrsistent._transformations import transform

def _is_pclass(bases):
    if False:
        print('Hello World!')
    return len(bases) == 1 and bases[0] == CheckedType

class PClassMeta(type):

    def __new__(mcs, name, bases, dct):
        if False:
            for i in range(10):
                print('nop')
        set_fields(dct, bases, name='_pclass_fields')
        store_invariants(dct, bases, '_pclass_invariants', '__invariant__')
        dct['__slots__'] = ('_pclass_frozen',) + tuple((key for key in dct['_pclass_fields']))
        if _is_pclass(bases):
            dct['__slots__'] += ('__weakref__',)
        return super(PClassMeta, mcs).__new__(mcs, name, bases, dct)
_MISSING_VALUE = object()

def _check_and_set_attr(cls, field, name, value, result, invariant_errors):
    if False:
        i = 10
        return i + 15
    check_type(cls, field, name, value)
    (is_ok, error_code) = field.invariant(value)
    if not is_ok:
        invariant_errors.append(error_code)
    else:
        setattr(result, name, value)

class PClass(CheckedType, metaclass=PClassMeta):
    """
    A PClass is a python class with a fixed set of specified fields. PClasses are declared as python classes inheriting
    from PClass. It is defined the same way that PRecords are and behaves like a PRecord in all aspects except that it
    is not a PMap and hence not a collection but rather a plain Python object.


    More documentation and examples of PClass usage is available at https://github.com/tobgu/pyrsistent
    """

    def __new__(cls, **kwargs):
        if False:
            i = 10
            return i + 15
        result = super(PClass, cls).__new__(cls)
        factory_fields = kwargs.pop('_factory_fields', None)
        ignore_extra = kwargs.pop('ignore_extra', None)
        missing_fields = []
        invariant_errors = []
        for (name, field) in cls._pclass_fields.items():
            if name in kwargs:
                if factory_fields is None or name in factory_fields:
                    if is_field_ignore_extra_complaint(PClass, field, ignore_extra):
                        value = field.factory(kwargs[name], ignore_extra=ignore_extra)
                    else:
                        value = field.factory(kwargs[name])
                else:
                    value = kwargs[name]
                _check_and_set_attr(cls, field, name, value, result, invariant_errors)
                del kwargs[name]
            elif field.initial is not PFIELD_NO_INITIAL:
                initial = field.initial() if callable(field.initial) else field.initial
                _check_and_set_attr(cls, field, name, initial, result, invariant_errors)
            elif field.mandatory:
                missing_fields.append('{0}.{1}'.format(cls.__name__, name))
        if invariant_errors or missing_fields:
            raise InvariantException(tuple(invariant_errors), tuple(missing_fields), 'Field invariant failed')
        if kwargs:
            raise AttributeError("'{0}' are not among the specified fields for {1}".format(', '.join(kwargs), cls.__name__))
        check_global_invariants(result, cls._pclass_invariants)
        result._pclass_frozen = True
        return result

    def set(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        "\n        Set a field in the instance. Returns a new instance with the updated value. The original instance remains\n        unmodified. Accepts key-value pairs or single string representing the field name and a value.\n\n        >>> from pyrsistent import PClass, field\n        >>> class AClass(PClass):\n        ...     x = field()\n        ...\n        >>> a = AClass(x=1)\n        >>> a2 = a.set(x=2)\n        >>> a3 = a.set('x', 3)\n        >>> a\n        AClass(x=1)\n        >>> a2\n        AClass(x=2)\n        >>> a3\n        AClass(x=3)\n        "
        if args:
            kwargs[args[0]] = args[1]
        factory_fields = set(kwargs)
        for key in self._pclass_fields:
            if key not in kwargs:
                value = getattr(self, key, _MISSING_VALUE)
                if value is not _MISSING_VALUE:
                    kwargs[key] = value
        return self.__class__(_factory_fields=factory_fields, **kwargs)

    @classmethod
    def create(cls, kwargs, _factory_fields=None, ignore_extra=False):
        if False:
            while True:
                i = 10
        '\n        Factory method. Will create a new PClass of the current type and assign the values\n        specified in kwargs.\n\n        :param ignore_extra: A boolean which when set to True will ignore any keys which appear in kwargs that are not\n                             in the set of fields on the PClass.\n        '
        if isinstance(kwargs, cls):
            return kwargs
        if ignore_extra:
            kwargs = {k: kwargs[k] for k in cls._pclass_fields if k in kwargs}
        return cls(_factory_fields=_factory_fields, ignore_extra=ignore_extra, **kwargs)

    def serialize(self, format=None):
        if False:
            return 10
        '\n        Serialize the current PClass using custom serializer functions for fields where\n        such have been supplied.\n        '
        result = {}
        for name in self._pclass_fields:
            value = getattr(self, name, _MISSING_VALUE)
            if value is not _MISSING_VALUE:
                result[name] = serialize(self._pclass_fields[name].serializer, format, value)
        return result

    def transform(self, *transformations):
        if False:
            while True:
                i = 10
        '\n        Apply transformations to the currency PClass. For more details on transformations see\n        the documentation for PMap. Transformations on PClasses do not support key matching\n        since the PClass is not a collection. Apart from that the transformations available\n        for other persistent types work as expected.\n        '
        return transform(self, transformations)

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(other, self.__class__):
            for name in self._pclass_fields:
                if getattr(self, name, _MISSING_VALUE) != getattr(other, name, _MISSING_VALUE):
                    return False
            return True
        return NotImplemented

    def __ne__(self, other):
        if False:
            return 10
        return not self == other

    def __hash__(self):
        if False:
            for i in range(10):
                print('nop')
        return hash(tuple(((key, getattr(self, key, _MISSING_VALUE)) for key in self._pclass_fields)))

    def __setattr__(self, key, value):
        if False:
            for i in range(10):
                print('nop')
        if getattr(self, '_pclass_frozen', False):
            raise AttributeError("Can't set attribute, key={0}, value={1}".format(key, value))
        super(PClass, self).__setattr__(key, value)

    def __delattr__(self, key):
        if False:
            i = 10
            return i + 15
        raise AttributeError("Can't delete attribute, key={0}, use remove()".format(key))

    def _to_dict(self):
        if False:
            i = 10
            return i + 15
        result = {}
        for key in self._pclass_fields:
            value = getattr(self, key, _MISSING_VALUE)
            if value is not _MISSING_VALUE:
                result[key] = value
        return result

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return '{0}({1})'.format(self.__class__.__name__, ', '.join(('{0}={1}'.format(k, repr(v)) for (k, v) in self._to_dict().items())))

    def __reduce__(self):
        if False:
            for i in range(10):
                print('nop')
        data = dict(((key, getattr(self, key)) for key in self._pclass_fields if hasattr(self, key)))
        return (_restore_pickle, (self.__class__, data))

    def evolver(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns an evolver for this object.\n        '
        return _PClassEvolver(self, self._to_dict())

    def remove(self, name):
        if False:
            while True:
                i = 10
        "\n        Remove attribute given by name from the current instance. Raises AttributeError if the\n        attribute doesn't exist.\n        "
        evolver = self.evolver()
        del evolver[name]
        return evolver.persistent()

class _PClassEvolver(object):
    __slots__ = ('_pclass_evolver_original', '_pclass_evolver_data', '_pclass_evolver_data_is_dirty', '_factory_fields')

    def __init__(self, original, initial_dict):
        if False:
            for i in range(10):
                print('nop')
        self._pclass_evolver_original = original
        self._pclass_evolver_data = initial_dict
        self._pclass_evolver_data_is_dirty = False
        self._factory_fields = set()

    def __getitem__(self, item):
        if False:
            print('Hello World!')
        return self._pclass_evolver_data[item]

    def set(self, key, value):
        if False:
            i = 10
            return i + 15
        if self._pclass_evolver_data.get(key, _MISSING_VALUE) is not value:
            self._pclass_evolver_data[key] = value
            self._factory_fields.add(key)
            self._pclass_evolver_data_is_dirty = True
        return self

    def __setitem__(self, key, value):
        if False:
            i = 10
            return i + 15
        self.set(key, value)

    def remove(self, item):
        if False:
            i = 10
            return i + 15
        if item in self._pclass_evolver_data:
            del self._pclass_evolver_data[item]
            self._factory_fields.discard(item)
            self._pclass_evolver_data_is_dirty = True
            return self
        raise AttributeError(item)

    def __delitem__(self, item):
        if False:
            return 10
        self.remove(item)

    def persistent(self):
        if False:
            i = 10
            return i + 15
        if self._pclass_evolver_data_is_dirty:
            return self._pclass_evolver_original.__class__(_factory_fields=self._factory_fields, **self._pclass_evolver_data)
        return self._pclass_evolver_original

    def __setattr__(self, key, value):
        if False:
            for i in range(10):
                print('nop')
        if key not in self.__slots__:
            self.set(key, value)
        else:
            super(_PClassEvolver, self).__setattr__(key, value)

    def __getattr__(self, item):
        if False:
            return 10
        return self[item]