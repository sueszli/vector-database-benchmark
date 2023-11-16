from pyrsistent._checked_types import CheckedType, _restore_pickle, InvariantException, store_invariants
from pyrsistent._field_common import set_fields, check_type, is_field_ignore_extra_complaint, PFIELD_NO_INITIAL, serialize, check_global_invariants
from pyrsistent._pmap import PMap, pmap

class _PRecordMeta(type):

    def __new__(mcs, name, bases, dct):
        if False:
            for i in range(10):
                print('nop')
        set_fields(dct, bases, name='_precord_fields')
        store_invariants(dct, bases, '_precord_invariants', '__invariant__')
        dct['_precord_mandatory_fields'] = set((name for (name, field) in dct['_precord_fields'].items() if field.mandatory))
        dct['_precord_initial_values'] = dict(((k, field.initial) for (k, field) in dct['_precord_fields'].items() if field.initial is not PFIELD_NO_INITIAL))
        dct['__slots__'] = ()
        return super(_PRecordMeta, mcs).__new__(mcs, name, bases, dct)

class PRecord(PMap, CheckedType, metaclass=_PRecordMeta):
    """
    A PRecord is a PMap with a fixed set of specified fields. Records are declared as python classes inheriting
    from PRecord. Because it is a PMap it has full support for all Mapping methods such as iteration and element
    access using subscript notation.

    More documentation and examples of PRecord usage is available at https://github.com/tobgu/pyrsistent
    """

    def __new__(cls, **kwargs):
        if False:
            print('Hello World!')
        if '_precord_size' in kwargs and '_precord_buckets' in kwargs:
            return super(PRecord, cls).__new__(cls, kwargs['_precord_size'], kwargs['_precord_buckets'])
        factory_fields = kwargs.pop('_factory_fields', None)
        ignore_extra = kwargs.pop('_ignore_extra', False)
        initial_values = kwargs
        if cls._precord_initial_values:
            initial_values = dict(((k, v() if callable(v) else v) for (k, v) in cls._precord_initial_values.items()))
            initial_values.update(kwargs)
        e = _PRecordEvolver(cls, pmap(pre_size=len(cls._precord_fields)), _factory_fields=factory_fields, _ignore_extra=ignore_extra)
        for (k, v) in initial_values.items():
            e[k] = v
        return e.persistent()

    def set(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Set a field in the record. This set function differs slightly from that in the PMap\n        class. First of all it accepts key-value pairs. Second it accepts multiple key-value\n        pairs to perform one, atomic, update of multiple fields.\n        '
        if args:
            return super(PRecord, self).set(args[0], args[1])
        return self.update(kwargs)

    def evolver(self):
        if False:
            while True:
                i = 10
        '\n        Returns an evolver of this object.\n        '
        return _PRecordEvolver(self.__class__, self)

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return '{0}({1})'.format(self.__class__.__name__, ', '.join(('{0}={1}'.format(k, repr(v)) for (k, v) in self.items())))

    @classmethod
    def create(cls, kwargs, _factory_fields=None, ignore_extra=False):
        if False:
            i = 10
            return i + 15
        '\n        Factory method. Will create a new PRecord of the current type and assign the values\n        specified in kwargs.\n\n        :param ignore_extra: A boolean which when set to True will ignore any keys which appear in kwargs that are not\n                             in the set of fields on the PRecord.\n        '
        if isinstance(kwargs, cls):
            return kwargs
        if ignore_extra:
            kwargs = {k: kwargs[k] for k in cls._precord_fields if k in kwargs}
        return cls(_factory_fields=_factory_fields, _ignore_extra=ignore_extra, **kwargs)

    def __reduce__(self):
        if False:
            return 10
        return (_restore_pickle, (self.__class__, dict(self)))

    def serialize(self, format=None):
        if False:
            while True:
                i = 10
        '\n        Serialize the current PRecord using custom serializer functions for fields where\n        such have been supplied.\n        '
        return dict(((k, serialize(self._precord_fields[k].serializer, format, v)) for (k, v) in self.items()))

class _PRecordEvolver(PMap._Evolver):
    __slots__ = ('_destination_cls', '_invariant_error_codes', '_missing_fields', '_factory_fields', '_ignore_extra')

    def __init__(self, cls, original_pmap, _factory_fields=None, _ignore_extra=False):
        if False:
            while True:
                i = 10
        super(_PRecordEvolver, self).__init__(original_pmap)
        self._destination_cls = cls
        self._invariant_error_codes = []
        self._missing_fields = []
        self._factory_fields = _factory_fields
        self._ignore_extra = _ignore_extra

    def __setitem__(self, key, original_value):
        if False:
            while True:
                i = 10
        self.set(key, original_value)

    def set(self, key, original_value):
        if False:
            i = 10
            return i + 15
        field = self._destination_cls._precord_fields.get(key)
        if field:
            if self._factory_fields is None or field in self._factory_fields:
                try:
                    if is_field_ignore_extra_complaint(PRecord, field, self._ignore_extra):
                        value = field.factory(original_value, ignore_extra=self._ignore_extra)
                    else:
                        value = field.factory(original_value)
                except InvariantException as e:
                    self._invariant_error_codes += e.invariant_errors
                    self._missing_fields += e.missing_fields
                    return self
            else:
                value = original_value
            check_type(self._destination_cls, field, key, value)
            (is_ok, error_code) = field.invariant(value)
            if not is_ok:
                self._invariant_error_codes.append(error_code)
            return super(_PRecordEvolver, self).set(key, value)
        else:
            raise AttributeError("'{0}' is not among the specified fields for {1}".format(key, self._destination_cls.__name__))

    def persistent(self):
        if False:
            print('Hello World!')
        cls = self._destination_cls
        is_dirty = self.is_dirty()
        pm = super(_PRecordEvolver, self).persistent()
        if is_dirty or not isinstance(pm, cls):
            result = cls(_precord_buckets=pm._buckets, _precord_size=pm._size)
        else:
            result = pm
        if cls._precord_mandatory_fields:
            self._missing_fields += tuple(('{0}.{1}'.format(cls.__name__, f) for f in cls._precord_mandatory_fields - set(result.keys())))
        if self._invariant_error_codes or self._missing_fields:
            raise InvariantException(tuple(self._invariant_error_codes), tuple(self._missing_fields), 'Field invariant failed')
        check_global_invariants(result, cls._precord_invariants)
        return result