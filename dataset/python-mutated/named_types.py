from __future__ import annotations
from typing import Any
from typing import Optional
from typing import Type
from typing import TYPE_CHECKING
from typing import Union
from ... import schema
from ... import util
from ...sql import coercions
from ...sql import elements
from ...sql import roles
from ...sql import sqltypes
from ...sql import type_api
from ...sql.base import _NoArg
from ...sql.ddl import InvokeCreateDDLBase
from ...sql.ddl import InvokeDropDDLBase
if TYPE_CHECKING:
    from ...sql._typing import _TypeEngineArgument

class NamedType(sqltypes.TypeEngine):
    """Base for named types."""
    __abstract__ = True
    DDLGenerator: Type[NamedTypeGenerator]
    DDLDropper: Type[NamedTypeDropper]
    create_type: bool

    def create(self, bind, checkfirst=True, **kw):
        if False:
            while True:
                i = 10
        'Emit ``CREATE`` DDL for this type.\n\n        :param bind: a connectable :class:`_engine.Engine`,\n         :class:`_engine.Connection`, or similar object to emit\n         SQL.\n        :param checkfirst: if ``True``, a query against\n         the PG catalog will be first performed to see\n         if the type does not exist already before\n         creating.\n\n        '
        bind._run_ddl_visitor(self.DDLGenerator, self, checkfirst=checkfirst)

    def drop(self, bind, checkfirst=True, **kw):
        if False:
            for i in range(10):
                print('nop')
        'Emit ``DROP`` DDL for this type.\n\n        :param bind: a connectable :class:`_engine.Engine`,\n         :class:`_engine.Connection`, or similar object to emit\n         SQL.\n        :param checkfirst: if ``True``, a query against\n         the PG catalog will be first performed to see\n         if the type actually exists before dropping.\n\n        '
        bind._run_ddl_visitor(self.DDLDropper, self, checkfirst=checkfirst)

    def _check_for_name_in_memos(self, checkfirst, kw):
        if False:
            return 10
        'Look in the \'ddl runner\' for \'memos\', then\n        note our name in that collection.\n\n        This to ensure a particular named type is operated\n        upon only once within any kind of create/drop\n        sequence without relying upon "checkfirst".\n\n        '
        if not self.create_type:
            return True
        if '_ddl_runner' in kw:
            ddl_runner = kw['_ddl_runner']
            type_name = f'pg_{self.__visit_name__}'
            if type_name in ddl_runner.memo:
                existing = ddl_runner.memo[type_name]
            else:
                existing = ddl_runner.memo[type_name] = set()
            present = (self.schema, self.name) in existing
            existing.add((self.schema, self.name))
            return present
        else:
            return False

    def _on_table_create(self, target, bind, checkfirst=False, **kw):
        if False:
            i = 10
            return i + 15
        if (checkfirst or (not self.metadata and (not kw.get('_is_metadata_operation', False)))) and (not self._check_for_name_in_memos(checkfirst, kw)):
            self.create(bind=bind, checkfirst=checkfirst)

    def _on_table_drop(self, target, bind, checkfirst=False, **kw):
        if False:
            return 10
        if not self.metadata and (not kw.get('_is_metadata_operation', False)) and (not self._check_for_name_in_memos(checkfirst, kw)):
            self.drop(bind=bind, checkfirst=checkfirst)

    def _on_metadata_create(self, target, bind, checkfirst=False, **kw):
        if False:
            i = 10
            return i + 15
        if not self._check_for_name_in_memos(checkfirst, kw):
            self.create(bind=bind, checkfirst=checkfirst)

    def _on_metadata_drop(self, target, bind, checkfirst=False, **kw):
        if False:
            print('Hello World!')
        if not self._check_for_name_in_memos(checkfirst, kw):
            self.drop(bind=bind, checkfirst=checkfirst)

class NamedTypeGenerator(InvokeCreateDDLBase):

    def __init__(self, dialect, connection, checkfirst=False, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(connection, **kwargs)
        self.checkfirst = checkfirst

    def _can_create_type(self, type_):
        if False:
            for i in range(10):
                print('nop')
        if not self.checkfirst:
            return True
        effective_schema = self.connection.schema_for_object(type_)
        return not self.connection.dialect.has_type(self.connection, type_.name, schema=effective_schema)

class NamedTypeDropper(InvokeDropDDLBase):

    def __init__(self, dialect, connection, checkfirst=False, **kwargs):
        if False:
            return 10
        super().__init__(connection, **kwargs)
        self.checkfirst = checkfirst

    def _can_drop_type(self, type_):
        if False:
            print('Hello World!')
        if not self.checkfirst:
            return True
        effective_schema = self.connection.schema_for_object(type_)
        return self.connection.dialect.has_type(self.connection, type_.name, schema=effective_schema)

class EnumGenerator(NamedTypeGenerator):

    def visit_enum(self, enum):
        if False:
            for i in range(10):
                print('nop')
        if not self._can_create_type(enum):
            return
        with self.with_ddl_events(enum):
            self.connection.execute(CreateEnumType(enum))

class EnumDropper(NamedTypeDropper):

    def visit_enum(self, enum):
        if False:
            while True:
                i = 10
        if not self._can_drop_type(enum):
            return
        with self.with_ddl_events(enum):
            self.connection.execute(DropEnumType(enum))

class ENUM(NamedType, type_api.NativeForEmulated, sqltypes.Enum):
    """PostgreSQL ENUM type.

    This is a subclass of :class:`_types.Enum` which includes
    support for PG's ``CREATE TYPE`` and ``DROP TYPE``.

    When the builtin type :class:`_types.Enum` is used and the
    :paramref:`.Enum.native_enum` flag is left at its default of
    True, the PostgreSQL backend will use a :class:`_postgresql.ENUM`
    type as the implementation, so the special create/drop rules
    will be used.

    The create/drop behavior of ENUM is necessarily intricate, due to the
    awkward relationship the ENUM type has in relationship to the
    parent table, in that it may be "owned" by just a single table, or
    may be shared among many tables.

    When using :class:`_types.Enum` or :class:`_postgresql.ENUM`
    in an "inline" fashion, the ``CREATE TYPE`` and ``DROP TYPE`` is emitted
    corresponding to when the :meth:`_schema.Table.create` and
    :meth:`_schema.Table.drop`
    methods are called::

        table = Table('sometable', metadata,
            Column('some_enum', ENUM('a', 'b', 'c', name='myenum'))
        )

        table.create(engine)  # will emit CREATE ENUM and CREATE TABLE
        table.drop(engine)  # will emit DROP TABLE and DROP ENUM

    To use a common enumerated type between multiple tables, the best
    practice is to declare the :class:`_types.Enum` or
    :class:`_postgresql.ENUM` independently, and associate it with the
    :class:`_schema.MetaData` object itself::

        my_enum = ENUM('a', 'b', 'c', name='myenum', metadata=metadata)

        t1 = Table('sometable_one', metadata,
            Column('some_enum', myenum)
        )

        t2 = Table('sometable_two', metadata,
            Column('some_enum', myenum)
        )

    When this pattern is used, care must still be taken at the level
    of individual table creates.  Emitting CREATE TABLE without also
    specifying ``checkfirst=True`` will still cause issues::

        t1.create(engine) # will fail: no such type 'myenum'

    If we specify ``checkfirst=True``, the individual table-level create
    operation will check for the ``ENUM`` and create if not exists::

        # will check if enum exists, and emit CREATE TYPE if not
        t1.create(engine, checkfirst=True)

    When using a metadata-level ENUM type, the type will always be created
    and dropped if either the metadata-wide create/drop is called::

        metadata.create_all(engine)  # will emit CREATE TYPE
        metadata.drop_all(engine)  # will emit DROP TYPE

    The type can also be created and dropped directly::

        my_enum.create(engine)
        my_enum.drop(engine)

    """
    native_enum = True
    DDLGenerator = EnumGenerator
    DDLDropper = EnumDropper

    def __init__(self, *enums, name: Union[str, _NoArg, None]=_NoArg.NO_ARG, create_type: bool=True, **kw):
        if False:
            for i in range(10):
                print('nop')
        'Construct an :class:`_postgresql.ENUM`.\n\n        Arguments are the same as that of\n        :class:`_types.Enum`, but also including\n        the following parameters.\n\n        :param create_type: Defaults to True.\n         Indicates that ``CREATE TYPE`` should be\n         emitted, after optionally checking for the\n         presence of the type, when the parent\n         table is being created; and additionally\n         that ``DROP TYPE`` is called when the table\n         is dropped.    When ``False``, no check\n         will be performed and no ``CREATE TYPE``\n         or ``DROP TYPE`` is emitted, unless\n         :meth:`~.postgresql.ENUM.create`\n         or :meth:`~.postgresql.ENUM.drop`\n         are called directly.\n         Setting to ``False`` is helpful\n         when invoking a creation scheme to a SQL file\n         without access to the actual database -\n         the :meth:`~.postgresql.ENUM.create` and\n         :meth:`~.postgresql.ENUM.drop` methods can\n         be used to emit SQL to a target bind.\n\n        '
        native_enum = kw.pop('native_enum', None)
        if native_enum is False:
            util.warn('the native_enum flag does not apply to the sqlalchemy.dialects.postgresql.ENUM datatype; this type always refers to ENUM.   Use sqlalchemy.types.Enum for non-native enum.')
        self.create_type = create_type
        if name is not _NoArg.NO_ARG:
            kw['name'] = name
        super().__init__(*enums, **kw)

    def coerce_compared_value(self, op, value):
        if False:
            print('Hello World!')
        super_coerced_type = super().coerce_compared_value(op, value)
        if super_coerced_type._type_affinity is type_api.STRINGTYPE._type_affinity:
            return self
        else:
            return super_coerced_type

    @classmethod
    def __test_init__(cls):
        if False:
            while True:
                i = 10
        return cls(name='name')

    @classmethod
    def adapt_emulated_to_native(cls, impl, **kw):
        if False:
            while True:
                i = 10
        'Produce a PostgreSQL native :class:`_postgresql.ENUM` from plain\n        :class:`.Enum`.\n\n        '
        kw.setdefault('validate_strings', impl.validate_strings)
        kw.setdefault('name', impl.name)
        kw.setdefault('schema', impl.schema)
        kw.setdefault('inherit_schema', impl.inherit_schema)
        kw.setdefault('metadata', impl.metadata)
        kw.setdefault('_create_events', False)
        kw.setdefault('values_callable', impl.values_callable)
        kw.setdefault('omit_aliases', impl._omit_aliases)
        kw.setdefault('_adapted_from', impl)
        if type_api._is_native_for_emulated(impl.__class__):
            kw.setdefault('create_type', impl.create_type)
        return cls(**kw)

    def create(self, bind=None, checkfirst=True):
        if False:
            while True:
                i = 10
        'Emit ``CREATE TYPE`` for this\n        :class:`_postgresql.ENUM`.\n\n        If the underlying dialect does not support\n        PostgreSQL CREATE TYPE, no action is taken.\n\n        :param bind: a connectable :class:`_engine.Engine`,\n         :class:`_engine.Connection`, or similar object to emit\n         SQL.\n        :param checkfirst: if ``True``, a query against\n         the PG catalog will be first performed to see\n         if the type does not exist already before\n         creating.\n\n        '
        if not bind.dialect.supports_native_enum:
            return
        super().create(bind, checkfirst=checkfirst)

    def drop(self, bind=None, checkfirst=True):
        if False:
            i = 10
            return i + 15
        'Emit ``DROP TYPE`` for this\n        :class:`_postgresql.ENUM`.\n\n        If the underlying dialect does not support\n        PostgreSQL DROP TYPE, no action is taken.\n\n        :param bind: a connectable :class:`_engine.Engine`,\n         :class:`_engine.Connection`, or similar object to emit\n         SQL.\n        :param checkfirst: if ``True``, a query against\n         the PG catalog will be first performed to see\n         if the type actually exists before dropping.\n\n        '
        if not bind.dialect.supports_native_enum:
            return
        super().drop(bind, checkfirst=checkfirst)

    def get_dbapi_type(self, dbapi):
        if False:
            while True:
                i = 10
        "dont return dbapi.STRING for ENUM in PostgreSQL, since that's\n        a different type"
        return None

class DomainGenerator(NamedTypeGenerator):

    def visit_DOMAIN(self, domain):
        if False:
            print('Hello World!')
        if not self._can_create_type(domain):
            return
        with self.with_ddl_events(domain):
            self.connection.execute(CreateDomainType(domain))

class DomainDropper(NamedTypeDropper):

    def visit_DOMAIN(self, domain):
        if False:
            return 10
        if not self._can_drop_type(domain):
            return
        with self.with_ddl_events(domain):
            self.connection.execute(DropDomainType(domain))

class DOMAIN(NamedType, sqltypes.SchemaType):
    """Represent the DOMAIN PostgreSQL type.

    A domain is essentially a data type with optional constraints
    that restrict the allowed set of values. E.g.::

        PositiveInt = DOMAIN(
            "pos_int", Integer, check="VALUE > 0", not_null=True
        )

        UsPostalCode = DOMAIN(
            "us_postal_code",
            Text,
            check="VALUE ~ '^\\d{5}$' OR VALUE ~ '^\\d{5}-\\d{4}$'"
        )

    See the `PostgreSQL documentation`__ for additional details

    __ https://www.postgresql.org/docs/current/sql-createdomain.html

    .. versionadded:: 2.0

    """
    DDLGenerator = DomainGenerator
    DDLDropper = DomainDropper
    __visit_name__ = 'DOMAIN'

    def __init__(self, name: str, data_type: _TypeEngineArgument[Any], *, collation: Optional[str]=None, default: Optional[Union[str, elements.TextClause]]=None, constraint_name: Optional[str]=None, not_null: Optional[bool]=None, check: Optional[str]=None, create_type: bool=True, **kw: Any):
        if False:
            print('Hello World!')
        "\n        Construct a DOMAIN.\n\n        :param name: the name of the domain\n        :param data_type: The underlying data type of the domain.\n          This can include array specifiers.\n        :param collation: An optional collation for the domain.\n          If no collation is specified, the underlying data type's default\n          collation is used. The underlying type must be collatable if\n          ``collation`` is specified.\n        :param default: The DEFAULT clause specifies a default value for\n          columns of the domain data type. The default should be a string\n          or a :func:`_expression.text` value.\n          If no default value is specified, then the default value is\n          the null value.\n        :param constraint_name: An optional name for a constraint.\n          If not specified, the backend generates a name.\n        :param not_null: Values of this domain are prevented from being null.\n          By default domain are allowed to be null. If not specified\n          no nullability clause will be emitted.\n        :param check: CHECK clause specify integrity constraint or test\n          which values of the domain must satisfy. A constraint must be\n          an expression producing a Boolean result that can use the key\n          word VALUE to refer to the value being tested.\n          Differently from PostgreSQL, only a single check clause is\n          currently allowed in SQLAlchemy.\n        :param schema: optional schema name\n        :param metadata: optional :class:`_schema.MetaData` object which\n         this :class:`_postgresql.DOMAIN` will be directly associated\n        :param create_type: Defaults to True.\n         Indicates that ``CREATE TYPE`` should be emitted, after optionally\n         checking for the presence of the type, when the parent table is\n         being created; and additionally that ``DROP TYPE`` is called\n         when the table is dropped.\n\n        "
        self.data_type = type_api.to_instance(data_type)
        self.default = default
        self.collation = collation
        self.constraint_name = constraint_name
        self.not_null = not_null
        if check is not None:
            check = coercions.expect(roles.DDLExpressionRole, check)
        self.check = check
        self.create_type = create_type
        super().__init__(name=name, **kw)

    @classmethod
    def __test_init__(cls):
        if False:
            for i in range(10):
                print('nop')
        return cls('name', sqltypes.Integer)

class CreateEnumType(schema._CreateDropBase):
    __visit_name__ = 'create_enum_type'

class DropEnumType(schema._CreateDropBase):
    __visit_name__ = 'drop_enum_type'

class CreateDomainType(schema._CreateDropBase):
    """Represent a CREATE DOMAIN statement."""
    __visit_name__ = 'create_domain_type'

class DropDomainType(schema._CreateDropBase):
    """Represent a DROP DOMAIN statement."""
    __visit_name__ = 'drop_domain_type'