import sys
from pyrsistent._checked_types import CheckedPMap, CheckedPSet, CheckedPVector, CheckedType, InvariantException, _restore_pickle, get_type, maybe_parse_user_type, maybe_parse_many_user_types
from pyrsistent._checked_types import optional as optional_type
from pyrsistent._checked_types import wrap_invariant
import inspect
PY2 = sys.version_info[0] < 3

def set_fields(dct, bases, name):
    if False:
        i = 10
        return i + 15
    dct[name] = dict(sum([list(b.__dict__.get(name, {}).items()) for b in bases], []))
    for (k, v) in list(dct.items()):
        if isinstance(v, _PField):
            dct[name][k] = v
            del dct[k]

def check_global_invariants(subject, invariants):
    if False:
        while True:
            i = 10
    error_codes = tuple((error_code for (is_ok, error_code) in (invariant(subject) for invariant in invariants) if not is_ok))
    if error_codes:
        raise InvariantException(error_codes, (), 'Global invariant failed')

def serialize(serializer, format, value):
    if False:
        i = 10
        return i + 15
    if isinstance(value, CheckedType) and serializer is PFIELD_NO_SERIALIZER:
        return value.serialize(format)
    return serializer(format, value)

def check_type(destination_cls, field, name, value):
    if False:
        print('Hello World!')
    if field.type and (not any((isinstance(value, get_type(t)) for t in field.type))):
        actual_type = type(value)
        message = 'Invalid type for field {0}.{1}, was {2}'.format(destination_cls.__name__, name, actual_type.__name__)
        raise PTypeError(destination_cls, name, field.type, actual_type, message)

def is_type_cls(type_cls, field_type):
    if False:
        while True:
            i = 10
    if type(field_type) is set:
        return True
    types = tuple(field_type)
    if len(types) == 0:
        return False
    return issubclass(get_type(types[0]), type_cls)

def is_field_ignore_extra_complaint(type_cls, field, ignore_extra):
    if False:
        print('Hello World!')
    if not ignore_extra:
        return False
    if not is_type_cls(type_cls, field.type):
        return False
    if PY2:
        return 'ignore_extra' in inspect.getargspec(field.factory).args
    else:
        return 'ignore_extra' in inspect.signature(field.factory).parameters

class _PField(object):
    __slots__ = ('type', 'invariant', 'initial', 'mandatory', '_factory', 'serializer')

    def __init__(self, type, invariant, initial, mandatory, factory, serializer):
        if False:
            return 10
        self.type = type
        self.invariant = invariant
        self.initial = initial
        self.mandatory = mandatory
        self._factory = factory
        self.serializer = serializer

    @property
    def factory(self):
        if False:
            for i in range(10):
                print('nop')
        if self._factory is PFIELD_NO_FACTORY and len(self.type) == 1:
            typ = get_type(tuple(self.type)[0])
            if issubclass(typ, CheckedType):
                return typ.create
        return self._factory
PFIELD_NO_TYPE = ()
PFIELD_NO_INVARIANT = lambda _: (True, None)
PFIELD_NO_FACTORY = lambda x: x
PFIELD_NO_INITIAL = object()
PFIELD_NO_SERIALIZER = lambda _, value: value

def field(type=PFIELD_NO_TYPE, invariant=PFIELD_NO_INVARIANT, initial=PFIELD_NO_INITIAL, mandatory=False, factory=PFIELD_NO_FACTORY, serializer=PFIELD_NO_SERIALIZER):
    if False:
        return 10
    '\n    Field specification factory for :py:class:`PRecord`.\n\n    :param type: a type or iterable with types that are allowed for this field\n    :param invariant: a function specifying an invariant that must hold for the field\n    :param initial: value of field if not specified when instantiating the record\n    :param mandatory: boolean specifying if the field is mandatory or not\n    :param factory: function called when field is set.\n    :param serializer: function that returns a serialized version of the field\n    '
    if isinstance(type, (list, set, tuple)):
        types = set(maybe_parse_many_user_types(type))
    else:
        types = set(maybe_parse_user_type(type))
    invariant_function = wrap_invariant(invariant) if invariant != PFIELD_NO_INVARIANT and callable(invariant) else invariant
    field = _PField(type=types, invariant=invariant_function, initial=initial, mandatory=mandatory, factory=factory, serializer=serializer)
    _check_field_parameters(field)
    return field

def _check_field_parameters(field):
    if False:
        i = 10
        return i + 15
    for t in field.type:
        if not isinstance(t, type) and (not isinstance(t, str)):
            raise TypeError('Type parameter expected, not {0}'.format(type(t)))
    if field.initial is not PFIELD_NO_INITIAL and (not callable(field.initial)) and field.type and (not any((isinstance(field.initial, t) for t in field.type))):
        raise TypeError('Initial has invalid type {0}'.format(type(field.initial)))
    if not callable(field.invariant):
        raise TypeError('Invariant must be callable')
    if not callable(field.factory):
        raise TypeError('Factory must be callable')
    if not callable(field.serializer):
        raise TypeError('Serializer must be callable')

class PTypeError(TypeError):
    """
    Raised when trying to assign a value with a type that doesn't match the declared type.

    Attributes:
    source_class -- The class of the record
    field -- Field name
    expected_types  -- Types allowed for the field
    actual_type -- The non matching type
    """

    def __init__(self, source_class, field, expected_types, actual_type, *args, **kwargs):
        if False:
            print('Hello World!')
        super(PTypeError, self).__init__(*args, **kwargs)
        self.source_class = source_class
        self.field = field
        self.expected_types = expected_types
        self.actual_type = actual_type
SEQ_FIELD_TYPE_SUFFIXES = {CheckedPVector: 'PVector', CheckedPSet: 'PSet'}
_seq_field_types = {}

def _restore_seq_field_pickle(checked_class, item_type, data):
    if False:
        return 10
    'Unpickling function for auto-generated PVec/PSet field types.'
    type_ = _seq_field_types[checked_class, item_type]
    return _restore_pickle(type_, data)

def _types_to_names(types):
    if False:
        return 10
    'Convert a tuple of types to a human-readable string.'
    return ''.join((get_type(typ).__name__.capitalize() for typ in types))

def _make_seq_field_type(checked_class, item_type):
    if False:
        for i in range(10):
            print('nop')
    'Create a subclass of the given checked class with the given item type.'
    type_ = _seq_field_types.get((checked_class, item_type))
    if type_ is not None:
        return type_

    class TheType(checked_class):
        __type__ = item_type

        def __reduce__(self):
            if False:
                for i in range(10):
                    print('nop')
            return (_restore_seq_field_pickle, (checked_class, item_type, list(self)))
    suffix = SEQ_FIELD_TYPE_SUFFIXES[checked_class]
    TheType.__name__ = _types_to_names(TheType._checked_types) + suffix
    _seq_field_types[checked_class, item_type] = TheType
    return TheType

def _sequence_field(checked_class, item_type, optional, initial):
    if False:
        for i in range(10):
            print('nop')
    '\n    Create checked field for either ``PSet`` or ``PVector``.\n\n    :param checked_class: ``CheckedPSet`` or ``CheckedPVector``.\n    :param item_type: The required type for the items in the set.\n    :param optional: If true, ``None`` can be used as a value for\n        this field.\n    :param initial: Initial value to pass to factory.\n\n    :return: A ``field`` containing a checked class.\n    '
    TheType = _make_seq_field_type(checked_class, item_type)
    if optional:

        def factory(argument, _factory_fields=None, ignore_extra=False):
            if False:
                return 10
            if argument is None:
                return None
            else:
                return TheType.create(argument, _factory_fields=_factory_fields, ignore_extra=ignore_extra)
    else:
        factory = TheType.create
    return field(type=optional_type(TheType) if optional else TheType, factory=factory, mandatory=True, initial=factory(initial))

def pset_field(item_type, optional=False, initial=()):
    if False:
        for i in range(10):
            print('nop')
    '\n    Create checked ``PSet`` field.\n\n    :param item_type: The required type for the items in the set.\n    :param optional: If true, ``None`` can be used as a value for\n        this field.\n    :param initial: Initial value to pass to factory if no value is given\n        for the field.\n\n    :return: A ``field`` containing a ``CheckedPSet`` of the given type.\n    '
    return _sequence_field(CheckedPSet, item_type, optional, initial)

def pvector_field(item_type, optional=False, initial=()):
    if False:
        for i in range(10):
            print('nop')
    '\n    Create checked ``PVector`` field.\n\n    :param item_type: The required type for the items in the vector.\n    :param optional: If true, ``None`` can be used as a value for\n        this field.\n    :param initial: Initial value to pass to factory if no value is given\n        for the field.\n\n    :return: A ``field`` containing a ``CheckedPVector`` of the given type.\n    '
    return _sequence_field(CheckedPVector, item_type, optional, initial)
_valid = lambda item: (True, '')
_pmap_field_types = {}

def _restore_pmap_field_pickle(key_type, value_type, data):
    if False:
        while True:
            i = 10
    'Unpickling function for auto-generated PMap field types.'
    type_ = _pmap_field_types[key_type, value_type]
    return _restore_pickle(type_, data)

def _make_pmap_field_type(key_type, value_type):
    if False:
        print('Hello World!')
    'Create a subclass of CheckedPMap with the given key and value types.'
    type_ = _pmap_field_types.get((key_type, value_type))
    if type_ is not None:
        return type_

    class TheMap(CheckedPMap):
        __key_type__ = key_type
        __value_type__ = value_type

        def __reduce__(self):
            if False:
                while True:
                    i = 10
            return (_restore_pmap_field_pickle, (self.__key_type__, self.__value_type__, dict(self)))
    TheMap.__name__ = '{0}To{1}PMap'.format(_types_to_names(TheMap._checked_key_types), _types_to_names(TheMap._checked_value_types))
    _pmap_field_types[key_type, value_type] = TheMap
    return TheMap

def pmap_field(key_type, value_type, optional=False, invariant=PFIELD_NO_INVARIANT):
    if False:
        for i in range(10):
            print('nop')
    '\n    Create a checked ``PMap`` field.\n\n    :param key: The required type for the keys of the map.\n    :param value: The required type for the values of the map.\n    :param optional: If true, ``None`` can be used as a value for\n        this field.\n    :param invariant: Pass-through to ``field``.\n\n    :return: A ``field`` containing a ``CheckedPMap``.\n    '
    TheMap = _make_pmap_field_type(key_type, value_type)
    if optional:

        def factory(argument):
            if False:
                for i in range(10):
                    print('nop')
            if argument is None:
                return None
            else:
                return TheMap.create(argument)
    else:
        factory = TheMap.create
    return field(mandatory=True, initial=TheMap(), type=optional_type(TheMap) if optional else TheMap, factory=factory, invariant=invariant)