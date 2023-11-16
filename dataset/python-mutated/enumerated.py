import re
from .types import _StringType
from ... import exc
from ... import sql
from ... import util
from ...sql import sqltypes

class ENUM(sqltypes.NativeForEmulated, sqltypes.Enum, _StringType):
    """MySQL ENUM type."""
    __visit_name__ = 'ENUM'
    native_enum = True

    def __init__(self, *enums, **kw):
        if False:
            return 10
        'Construct an ENUM.\n\n        E.g.::\n\n          Column(\'myenum\', ENUM("foo", "bar", "baz"))\n\n        :param enums: The range of valid values for this ENUM.  Values in\n          enums are not quoted, they will be escaped and surrounded by single\n          quotes when generating the schema.  This object may also be a\n          PEP-435-compliant enumerated type.\n\n          .. versionadded: 1.1 added support for PEP-435-compliant enumerated\n             types.\n\n        :param strict: This flag has no effect.\n\n         .. versionchanged:: The MySQL ENUM type as well as the base Enum\n            type now validates all Python data values.\n\n        :param charset: Optional, a column-level character set for this string\n          value.  Takes precedence to \'ascii\' or \'unicode\' short-hand.\n\n        :param collation: Optional, a column-level collation for this string\n          value.  Takes precedence to \'binary\' short-hand.\n\n        :param ascii: Defaults to False: short-hand for the ``latin1``\n          character set, generates ASCII in schema.\n\n        :param unicode: Defaults to False: short-hand for the ``ucs2``\n          character set, generates UNICODE in schema.\n\n        :param binary: Defaults to False: short-hand, pick the binary\n          collation type that matches the column\'s character set.  Generates\n          BINARY in schema.  This does not affect the type of data stored,\n          only the collation of character data.\n\n        '
        kw.pop('strict', None)
        self._enum_init(enums, kw)
        _StringType.__init__(self, length=self.length, **kw)

    @classmethod
    def adapt_emulated_to_native(cls, impl, **kw):
        if False:
            while True:
                i = 10
        'Produce a MySQL native :class:`.mysql.ENUM` from plain\n        :class:`.Enum`.\n\n        '
        kw.setdefault('validate_strings', impl.validate_strings)
        kw.setdefault('values_callable', impl.values_callable)
        kw.setdefault('omit_aliases', impl._omit_aliases)
        return cls(**kw)

    def _object_value_for_elem(self, elem):
        if False:
            while True:
                i = 10
        if elem == '':
            return elem
        else:
            return super()._object_value_for_elem(elem)

    def __repr__(self):
        if False:
            return 10
        return util.generic_repr(self, to_inspect=[ENUM, _StringType, sqltypes.Enum])

class SET(_StringType):
    """MySQL SET type."""
    __visit_name__ = 'SET'

    def __init__(self, *values, **kw):
        if False:
            return 10
        'Construct a SET.\n\n        E.g.::\n\n          Column(\'myset\', SET("foo", "bar", "baz"))\n\n\n        The list of potential values is required in the case that this\n        set will be used to generate DDL for a table, or if the\n        :paramref:`.SET.retrieve_as_bitwise` flag is set to True.\n\n        :param values: The range of valid values for this SET. The values\n          are not quoted, they will be escaped and surrounded by single\n          quotes when generating the schema.\n\n        :param convert_unicode: Same flag as that of\n         :paramref:`.String.convert_unicode`.\n\n        :param collation: same as that of :paramref:`.String.collation`\n\n        :param charset: same as that of :paramref:`.VARCHAR.charset`.\n\n        :param ascii: same as that of :paramref:`.VARCHAR.ascii`.\n\n        :param unicode: same as that of :paramref:`.VARCHAR.unicode`.\n\n        :param binary: same as that of :paramref:`.VARCHAR.binary`.\n\n        :param retrieve_as_bitwise: if True, the data for the set type will be\n          persisted and selected using an integer value, where a set is coerced\n          into a bitwise mask for persistence.  MySQL allows this mode which\n          has the advantage of being able to store values unambiguously,\n          such as the blank string ``\'\'``.   The datatype will appear\n          as the expression ``col + 0`` in a SELECT statement, so that the\n          value is coerced into an integer value in result sets.\n          This flag is required if one wishes\n          to persist a set that can store the blank string ``\'\'`` as a value.\n\n          .. warning::\n\n            When using :paramref:`.mysql.SET.retrieve_as_bitwise`, it is\n            essential that the list of set values is expressed in the\n            **exact same order** as exists on the MySQL database.\n\n        '
        self.retrieve_as_bitwise = kw.pop('retrieve_as_bitwise', False)
        self.values = tuple(values)
        if not self.retrieve_as_bitwise and '' in values:
            raise exc.ArgumentError("Can't use the blank value '' in a SET without setting retrieve_as_bitwise=True")
        if self.retrieve_as_bitwise:
            self._bitmap = {value: 2 ** idx for (idx, value) in enumerate(self.values)}
            self._bitmap.update(((2 ** idx, value) for (idx, value) in enumerate(self.values)))
        length = max([len(v) for v in values] + [0])
        kw.setdefault('length', length)
        super().__init__(**kw)

    def column_expression(self, colexpr):
        if False:
            i = 10
            return i + 15
        if self.retrieve_as_bitwise:
            return sql.type_coerce(sql.type_coerce(colexpr, sqltypes.Integer) + 0, self)
        else:
            return colexpr

    def result_processor(self, dialect, coltype):
        if False:
            while True:
                i = 10
        if self.retrieve_as_bitwise:

            def process(value):
                if False:
                    while True:
                        i = 10
                if value is not None:
                    value = int(value)
                    return set(util.map_bits(self._bitmap.__getitem__, value))
                else:
                    return None
        else:
            super_convert = super().result_processor(dialect, coltype)

            def process(value):
                if False:
                    while True:
                        i = 10
                if isinstance(value, str):
                    if super_convert:
                        value = super_convert(value)
                    return set(re.findall('[^,]+', value))
                else:
                    if value is not None:
                        value.discard('')
                    return value
        return process

    def bind_processor(self, dialect):
        if False:
            while True:
                i = 10
        super_convert = super().bind_processor(dialect)
        if self.retrieve_as_bitwise:

            def process(value):
                if False:
                    return 10
                if value is None:
                    return None
                elif isinstance(value, (int, str)):
                    if super_convert:
                        return super_convert(value)
                    else:
                        return value
                else:
                    int_value = 0
                    for v in value:
                        int_value |= self._bitmap[v]
                    return int_value
        else:

            def process(value):
                if False:
                    return 10
                if value is not None and (not isinstance(value, (int, str))):
                    value = ','.join(value)
                if super_convert:
                    return super_convert(value)
                else:
                    return value
        return process

    def adapt(self, impltype, **kw):
        if False:
            while True:
                i = 10
        kw['retrieve_as_bitwise'] = self.retrieve_as_bitwise
        return util.constructor_copy(self, impltype, *self.values, **kw)

    def __repr__(self):
        if False:
            print('Hello World!')
        return util.generic_repr(self, to_inspect=[SET, _StringType], additional_kw=[('retrieve_as_bitwise', False)])