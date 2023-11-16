from __future__ import annotations
from typing import Any
from typing import Dict
from typing import Optional
from typing import Sequence
from typing import Tuple
from apache_beam.typehints import typehints
from apache_beam.typehints.native_type_compatibility import match_is_named_tuple
from apache_beam.typehints.schema_registry import SchemaTypeRegistry
_BEAM_SCHEMA_ID = '_beam_schema_id'

def _user_type_is_generated(user_type: type) -> bool:
    if False:
        i = 10
        return i + 15
    if not hasattr(user_type, _BEAM_SCHEMA_ID):
        return False
    schema_id = getattr(user_type, _BEAM_SCHEMA_ID)
    type_name = 'BeamSchema_{}'.format(schema_id.replace('-', '_'))
    return user_type.__name__ == type_name

class RowTypeConstraint(typehints.TypeConstraint):

    def __init__(self, fields: Sequence[Tuple[str, type]], user_type, schema_options: Optional[Sequence[Tuple[str, Any]]]=None, field_options: Optional[Dict[str, Sequence[Tuple[str, Any]]]]=None):
        if False:
            print('Hello World!')
        'For internal use only, no backwards comatibility guaratees.  See\n    https://beam.apache.org/documentation/programming-guide/#schemas-for-pl-types\n    for guidance on creating PCollections with inferred schemas.\n\n    Note RowTypeConstraint does not currently store arbitrary functions for\n    converting to/from the user type. Instead, we only support ``NamedTuple``\n    user types and make the follow assumptions:\n\n    - The user type can be constructed with field values as arguments in order\n      (i.e. ``constructor(*field_values)``).\n    - Field values can be accessed from instances of the user type by attribute\n      (i.e. with ``getattr(obj, field_name)``).\n\n    In the future we will add support for dataclasses\n    ([#22085](https://github.com/apache/beam/issues/22085)) which also satisfy\n    these assumptions.\n\n    The RowTypeConstraint constructor should not be called directly (even\n    internally to Beam). Prefer static methods ``from_user_type`` or\n    ``from_fields``.\n\n    Parameters:\n      fields: a list of (name, type) tuples, representing the schema inferred\n        from user_type.\n      user_type: constructor for a user type (e.g. NamedTuple class) that is\n        used to represent this schema in user code.\n      schema_options: A list of (key, value) tuples representing schema-level\n        options.\n      field_options: A dictionary representing field-level options. Dictionary\n        keys are field names, and dictionary values are lists of (key, value)\n        tuples representing field-level options for that field.\n    '
        self._fields = tuple(((name, RowTypeConstraint.from_user_type(typ) or typ) for (name, typ) in fields))
        self._user_type = user_type
        self._schema_id = getattr(self._user_type, _BEAM_SCHEMA_ID, None)
        self._schema_options = schema_options or []
        self._field_options = field_options or {}

    @staticmethod
    def from_user_type(user_type: type, schema_options: Optional[Sequence[Tuple[str, Any]]]=None, field_options: Optional[Dict[str, Sequence[Tuple[str, Any]]]]=None) -> Optional[RowTypeConstraint]:
        if False:
            return 10
        if match_is_named_tuple(user_type):
            fields = [(name, user_type.__annotations__[name]) for name in user_type._fields]
            if _user_type_is_generated(user_type):
                return RowTypeConstraint.from_fields(fields, schema_id=getattr(user_type, _BEAM_SCHEMA_ID), schema_options=schema_options, field_options=field_options)
            return RowTypeConstraint(fields=fields, user_type=user_type, schema_options=schema_options, field_options=field_options)
        return None

    @staticmethod
    def from_fields(fields: Sequence[Tuple[str, type]], schema_id: Optional[str]=None, schema_options: Optional[Sequence[Tuple[str, Any]]]=None, field_options: Optional[Dict[str, Sequence[Tuple[str, Any]]]]=None, schema_registry: Optional[SchemaTypeRegistry]=None) -> RowTypeConstraint:
        if False:
            return 10
        return GeneratedClassRowTypeConstraint(fields, schema_id=schema_id, schema_options=schema_options, field_options=field_options, schema_registry=schema_registry)

    def __call__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        return self._user_type(*args, **kwargs)

    @property
    def user_type(self):
        if False:
            print('Hello World!')
        return self._user_type

    def set_schema_id(self, schema_id):
        if False:
            for i in range(10):
                print('nop')
        self._schema_id = schema_id
        if self._user_type is not None:
            setattr(self._user_type, _BEAM_SCHEMA_ID, self._schema_id)

    @property
    def schema_id(self):
        if False:
            return 10
        return self._schema_id

    @property
    def schema_options(self):
        if False:
            i = 10
            return i + 15
        return self._schema_options

    def field_options(self, field_name):
        if False:
            print('Hello World!')
        return self._field_options.get(field_name, [])

    def _consistent_with_check_(self, sub):
        if False:
            while True:
                i = 10
        return self == sub

    def type_check(self, instance):
        if False:
            i = 10
            return i + 15
        from apache_beam import Row
        return isinstance(instance, (Row, self._user_type))

    def _inner_types(self):
        if False:
            return 10
        'Iterates over the inner types of the composite type.'
        return [field[1] for field in self._fields]

    def __eq__(self, other):
        if False:
            while True:
                i = 10
        return type(self) == type(other) and self._fields == other._fields

    def __hash__(self):
        if False:
            i = 10
            return i + 15
        return hash(self._fields)

    def __repr__(self):
        if False:
            return 10
        return 'Row(%s)' % ', '.join(('%s=%s' % (name, repr(t)) for (name, t) in self._fields))

    def get_type_for(self, name):
        if False:
            i = 10
            return i + 15
        return dict(self._fields)[name]

class GeneratedClassRowTypeConstraint(RowTypeConstraint):
    """Specialization of RowTypeConstraint which relies on a generated user_type.

  Since the generated user_type cannot be pickled, we supply a custom __reduce__
  function that will regenerate the user_type.
  """

    def __init__(self, fields, schema_id: Optional[str]=None, schema_options: Optional[Sequence[Tuple[str, Any]]]=None, field_options: Optional[Dict[str, Sequence[Tuple[str, Any]]]]=None, schema_registry: Optional[SchemaTypeRegistry]=None):
        if False:
            print('Hello World!')
        from apache_beam.typehints.schemas import named_fields_to_schema
        from apache_beam.typehints.schemas import named_tuple_from_schema
        kwargs = {'schema_registry': schema_registry} if schema_registry else {}
        schema = named_fields_to_schema(fields, schema_id=schema_id, schema_options=schema_options, field_options=field_options, **kwargs)
        user_type = named_tuple_from_schema(schema, **kwargs)
        super().__init__(fields, user_type, schema_options=schema_options, field_options=field_options)

    def __reduce__(self):
        if False:
            for i in range(10):
                print('nop')
        return (RowTypeConstraint.from_fields, (self._fields, self._schema_id, self._schema_options, self._field_options, None))