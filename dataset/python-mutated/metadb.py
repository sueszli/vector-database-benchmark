"""
A SQLAlchemy dialect for querying across Superset databases.

The dialect ``superset://`` allows users to query any table in any database that has been
configured in Superset, eg:

    > SELECT * FROM "examples.birth_names";

The syntax for tables is:

    database[[.catalog].schema].table

The dialect is built on top of Shillelagh, a framework for building DB API 2.0 libraries
and SQLAlchemy dialects based on SQLite. SQLite will parse the SQL, and pass the filters
to the adapter. The adapter builds a SQLAlchemy query object reading data from the table
and applying any filters (as well as sorting, limiting, and offsetting).

Note that no aggregation is done on the database. Aggregations and other operations like
joins and unions are done in memory, using the SQLite engine.
"""
from __future__ import annotations
import datetime
import operator
import urllib.parse
from collections.abc import Iterator
from functools import partial, wraps
from typing import Any, Callable, cast, TypeVar
from flask import current_app
from shillelagh.adapters.base import Adapter
from shillelagh.backends.apsw.dialects.base import APSWDialect
from shillelagh.exceptions import ProgrammingError
from shillelagh.fields import Blob, Boolean, Date, DateTime, Field, Float, Integer, Order, String, Time
from shillelagh.filters import Equal, Filter, Range
from shillelagh.typing import RequestedOrder, Row
from sqlalchemy import func, MetaData, Table
from sqlalchemy.engine.url import URL
from sqlalchemy.exc import NoSuchTableError
from sqlalchemy.sql import Select, select
from superset import db, feature_flag_manager, security_manager, sql_parse

class SupersetAPSWDialect(APSWDialect):
    """
    A SQLAlchemy dialect for an internal Superset engine.

    This dialect allows query to be executed across different Superset
    databases. For example, to read data from the `birth_names` table in the
    `examples` databases:

        >>> engine = create_engine('superset://')
        >>> conn = engine.connect()
        >>> results = conn.execute('SELECT * FROM "examples.birth_names"')

    Queries can also join data across different Superset databases.

    The dialect is built in top of the shillelagh library, leveraging SQLite to
    create virtual tables on-the-fly proxying Superset tables. The
    `SupersetShillelaghAdapter` adapter is responsible for returning data when a
    Superset table is accessed.
    """
    name = 'superset'

    def __init__(self, allowed_dbs: list[str] | None=None, **kwargs: Any) -> None:
        if False:
            print('Hello World!')
        super().__init__(**kwargs)
        self.allowed_dbs = allowed_dbs

    def create_connect_args(self, url: URL) -> tuple[tuple[()], dict[str, Any]]:
        if False:
            while True:
                i = 10
        '\n        A custom Shillelagh SQLAlchemy dialect with a single adapter configured.\n        '
        return ((), {'path': ':memory:', 'adapters': ['superset'], 'adapter_kwargs': {'superset': {'prefix': None, 'allowed_dbs': self.allowed_dbs}}, 'safe': True, 'isolation_level': self.isolation_level})
F = TypeVar('F', bound=Callable[..., Any])

def check_dml(method: F) -> F:
    if False:
        for i in range(10):
            print('nop')
    "\n    Decorator that prevents DML against databases where it's not allowed.\n    "

    @wraps(method)
    def wrapper(self: SupersetShillelaghAdapter, *args: Any, **kwargs: Any) -> Any:
        if False:
            return 10
        if not self._allow_dml:
            raise ProgrammingError(f'DML not enabled in database "{self.database}"')
        return method(self, *args, **kwargs)
    return cast(F, wrapper)

def has_rowid(method: F) -> F:
    if False:
        i = 10
        return i + 15
    '\n    Decorator that prevents updates/deletes on tables without a rowid.\n    '

    @wraps(method)
    def wrapper(self: SupersetShillelaghAdapter, *args: Any, **kwargs: Any) -> Any:
        if False:
            print('Hello World!')
        if not self._rowid:
            raise ProgrammingError('Can only modify data in a table with a single, integer, primary key')
        return method(self, *args, **kwargs)
    return cast(F, wrapper)

class Duration(Field[datetime.timedelta, datetime.timedelta]):
    """
    Shillelagh field used for representing durations as `timedelta` objects.
    """
    type = 'DURATION'
    db_api_type = 'DATETIME'

class SupersetShillelaghAdapter(Adapter):
    """
    A shillelagh adapter for Superset tables.

    Shillelagh adapters are responsible for fetching data from a given resource,
    allowing it to be represented as a virtual table in SQLite. This one works
    as a proxy to Superset tables.
    """
    safe = True
    supports_limit = True
    supports_offset = True
    type_map: dict[Any, type[Field]] = {bool: Boolean, float: Float, int: Integer, str: String, datetime.date: Date, datetime.datetime: DateTime, datetime.time: Time, datetime.timedelta: Duration}

    @staticmethod
    def supports(uri: str, fast: bool=True, prefix: str | None='superset', allowed_dbs: list[str] | None=None, **kwargs: Any) -> bool:
        if False:
            i = 10
            return i + 15
        '\n        Return if a table is supported by the adapter.\n\n        An URL for a table has the format [prefix.]database[[.catalog].schema].table,\n        eg, `superset.examples.birth_names`.\n\n        When using the Superset SQLAlchemy and DB engine spec the prefix is dropped, so\n        that tables should have the format database[[.catalog].schema].table.\n        '
        parts = [urllib.parse.unquote(part) for part in uri.split('.')]
        if prefix is not None:
            if parts.pop(0) != prefix:
                return False
        if allowed_dbs is not None and parts[0] not in allowed_dbs:
            return False
        return 2 <= len(parts) <= 4

    @staticmethod
    def parse_uri(uri: str) -> tuple[str]:
        if False:
            print('Hello World!')
        '\n        Pass URI through unmodified.\n        '
        return (uri,)

    def __init__(self, uri: str, prefix: str | None='superset', **kwargs: Any):
        if False:
            while True:
                i = 10
        if not feature_flag_manager.is_feature_enabled('ENABLE_SUPERSET_META_DB'):
            raise ProgrammingError('Superset meta database is disabled')
        super().__init__(**kwargs)
        parts = [urllib.parse.unquote(part) for part in uri.split('.')]
        if prefix is not None:
            if prefix != parts.pop(0):
                raise ProgrammingError('Invalid prefix')
            self.prefix = prefix
        self.database = parts.pop(0)
        self.table = parts.pop(-1)
        self.schema = parts.pop(-1) if parts else None
        self.catalog = parts.pop(-1) if parts else None
        if self.catalog:
            raise NotImplementedError('Catalogs are not currently supported')
        self._rowid: str | None = None
        self._allow_dml: bool = False
        self._set_columns()

    @classmethod
    def get_field(cls, python_type: Any) -> Field:
        if False:
            i = 10
            return i + 15
        '\n        Convert a Python type into a Shillelagh field.\n        '
        class_ = cls.type_map.get(python_type, Blob)
        return class_(filters=[Equal, Range], order=Order.ANY, exact=True)

    def _set_columns(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        "\n        Inspect the table and get its columns.\n\n        This is done on initialization because it's expensive.\n        "
        from superset.models.core import Database
        database = db.session.query(Database).filter_by(database_name=self.database).first()
        if database is None:
            raise ProgrammingError(f'Database not found: {self.database}')
        self._allow_dml = database.allow_dml
        table = sql_parse.Table(self.table, self.schema, self.catalog)
        security_manager.raise_for_access(database=database, table=table)
        self.engine_context = partial(database.get_sqla_engine_with_context, self.schema)
        metadata = MetaData()
        with self.engine_context() as engine:
            try:
                self._table = Table(self.table, metadata, schema=self.schema, autoload=True, autoload_with=engine)
            except NoSuchTableError as ex:
                raise ProgrammingError(f'Table does not exist: {self.table}') from ex
        primary_keys = [column for column in list(self._table.primary_key) if column.primary_key]
        if len(primary_keys) == 1 and primary_keys[0].type.python_type == int:
            self._rowid = primary_keys[0].name
        self.columns = {column.name: self.get_field(column.type.python_type) for column in self._table.c}

    def get_columns(self) -> dict[str, Field]:
        if False:
            i = 10
            return i + 15
        '\n        Return table columns.\n        '
        return self.columns

    def _build_sql(self, bounds: dict[str, Filter], order: list[tuple[str, RequestedOrder]], limit: int | None=None, offset: int | None=None) -> Select:
        if False:
            print('Hello World!')
        '\n        Build SQLAlchemy query object.\n        '
        query = select([self._table])
        for (column_name, filter_) in bounds.items():
            column = self._table.c[column_name]
            if isinstance(filter_, Equal):
                query = query.where(column == filter_.value)
            elif isinstance(filter_, Range):
                if filter_.start is not None:
                    op = operator.ge if filter_.include_start else operator.gt
                    query = query.where(op(column, filter_.start))
                if filter_.end is not None:
                    op = operator.le if filter_.include_end else operator.lt
                    query = query.where(op(column, filter_.end))
            else:
                raise ProgrammingError(f'Invalid filter: {filter_}')
        for (column_name, requested_order) in order:
            column = self._table.c[column_name]
            if requested_order == Order.DESCENDING:
                column = column.desc()
            query = query.order_by(column)
        if limit is not None:
            query = query.limit(limit)
        if offset is not None:
            query = query.offset(offset)
        return query

    def get_data(self, bounds: dict[str, Filter], order: list[tuple[str, RequestedOrder]], limit: int | None=None, offset: int | None=None, **kwargs: Any) -> Iterator[Row]:
        if False:
            return 10
        '\n        Return data for a `SELECT` statement.\n        '
        app_limit: int | None = current_app.config['SUPERSET_META_DB_LIMIT']
        if limit is None:
            limit = app_limit
        elif app_limit is not None:
            limit = min(limit, app_limit)
        query = self._build_sql(bounds, order, limit, offset)
        with self.engine_context() as engine:
            connection = engine.connect()
            rows = connection.execute(query)
            for (i, row) in enumerate(rows):
                data = dict(zip(self.columns, row))
                data['rowid'] = data[self._rowid] if self._rowid else i
                yield data

    @check_dml
    def insert_row(self, row: Row) -> int:
        if False:
            for i in range(10):
                print('nop')
        '\n        Insert a single row.\n        '
        row_id: int | None = row.pop('rowid')
        if row_id and self._rowid:
            if row.get(self._rowid) != row_id:
                raise ProgrammingError(f'Invalid rowid specified: {row_id}')
            row[self._rowid] = row_id
        if self._rowid and row[self._rowid] is None and self._table.c[self._rowid].autoincrement:
            row.pop(self._rowid)
        query = self._table.insert().values(**row)
        with self.engine_context() as engine:
            connection = engine.connect()
            result = connection.execute(query)
            if self._rowid:
                return result.inserted_primary_key[0]
            query = select([func.count()]).select_from(self._table)
            return connection.execute(query).scalar()

    @check_dml
    @has_rowid
    def delete_row(self, row_id: int) -> None:
        if False:
            while True:
                i = 10
        '\n        Delete a single row given its row ID.\n        '
        query = self._table.delete().where(self._table.c[self._rowid] == row_id)
        with self.engine_context() as engine:
            connection = engine.connect()
            connection.execute(query)

    @check_dml
    @has_rowid
    def update_row(self, row_id: int, row: Row) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Update a single row given its row ID.\n\n        Note that the updated row might have a new row ID.\n        '
        new_row_id: int | None = row.pop('rowid')
        if new_row_id:
            if row.get(self._rowid) != new_row_id:
                raise ProgrammingError(f'Invalid rowid specified: {new_row_id}')
            row[self._rowid] = new_row_id
        query = self._table.update().where(self._table.c[self._rowid] == row_id).values(**row)
        with self.engine_context() as engine:
            connection = engine.connect()
            connection.execute(query)