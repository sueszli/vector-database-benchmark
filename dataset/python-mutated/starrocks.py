import logging
import re
from re import Pattern
from typing import Any, Optional
from urllib import parse
from flask_babel import gettext as __
from sqlalchemy import Float, Integer, Numeric, types
from sqlalchemy.engine.url import URL
from sqlalchemy.sql.type_api import TypeEngine
from superset.db_engine_specs.mysql import MySQLEngineSpec
from superset.errors import SupersetErrorType
from superset.utils.core import GenericDataType
CONNECTION_ACCESS_DENIED_REGEX = re.compile("Access denied for user '(?P<username>.*?)'")
CONNECTION_UNKNOWN_DATABASE_REGEX = re.compile("Unknown database '(?P<database>.*?)'")
logger = logging.getLogger(__name__)

class TINYINT(Integer):
    __visit_name__ = 'TINYINT'

class LARGEINT(Integer):
    __visit_name__ = 'LARGEINT'

class DOUBLE(Float):
    __visit_name__ = 'DOUBLE'

class HLL(Numeric):
    __visit_name__ = 'HLL'

class BITMAP(Numeric):
    __visit_name__ = 'BITMAP'

class PERCENTILE(Numeric):
    __visit_name__ = 'PERCENTILE'

class ARRAY(TypeEngine):
    __visit_name__ = 'ARRAY'

    @property
    def python_type(self) -> Optional[type[list[Any]]]:
        if False:
            for i in range(10):
                print('nop')
        return list

class MAP(TypeEngine):
    __visit_name__ = 'MAP'

    @property
    def python_type(self) -> Optional[type[dict[Any, Any]]]:
        if False:
            while True:
                i = 10
        return dict

class STRUCT(TypeEngine):
    __visit_name__ = 'STRUCT'

    @property
    def python_type(self) -> Optional[type[Any]]:
        if False:
            while True:
                i = 10
        return None

class StarRocksEngineSpec(MySQLEngineSpec):
    engine = 'starrocks'
    engine_name = 'StarRocks'
    default_driver = 'starrocks'
    sqlalchemy_uri_placeholder = 'starrocks://user:password@host:port/catalog.db[?key=value&key=value...]'
    column_type_mappings = ((re.compile('^tinyint', re.IGNORECASE), TINYINT(), GenericDataType.NUMERIC), (re.compile('^largeint', re.IGNORECASE), LARGEINT(), GenericDataType.NUMERIC), (re.compile('^decimal.*', re.IGNORECASE), types.DECIMAL(), GenericDataType.NUMERIC), (re.compile('^double', re.IGNORECASE), DOUBLE(), GenericDataType.NUMERIC), (re.compile('^varchar(\\((\\d+)\\))*$', re.IGNORECASE), types.VARCHAR(), GenericDataType.STRING), (re.compile('^char(\\((\\d+)\\))*$', re.IGNORECASE), types.CHAR(), GenericDataType.STRING), (re.compile('^json', re.IGNORECASE), types.JSON(), GenericDataType.STRING), (re.compile('^binary.*', re.IGNORECASE), types.String(), GenericDataType.STRING), (re.compile('^percentile', re.IGNORECASE), PERCENTILE(), GenericDataType.STRING), (re.compile('^hll', re.IGNORECASE), HLL(), GenericDataType.STRING), (re.compile('^bitmap', re.IGNORECASE), BITMAP(), GenericDataType.STRING), (re.compile('^array.*', re.IGNORECASE), ARRAY(), GenericDataType.STRING), (re.compile('^map.*', re.IGNORECASE), MAP(), GenericDataType.STRING), (re.compile('^struct.*', re.IGNORECASE), STRUCT(), GenericDataType.STRING))
    custom_errors: dict[Pattern[str], tuple[str, SupersetErrorType, dict[str, Any]]] = {CONNECTION_ACCESS_DENIED_REGEX: (__('Either the username "%(username)s" or the password is incorrect.'), SupersetErrorType.CONNECTION_ACCESS_DENIED_ERROR, {'invalid': ['username', 'password']}), CONNECTION_UNKNOWN_DATABASE_REGEX: (__('Unable to connect to database "%(database)s".'), SupersetErrorType.CONNECTION_UNKNOWN_DATABASE_ERROR, {'invalid': ['database']})}

    @classmethod
    def adjust_engine_params(cls, uri: URL, connect_args: dict[str, Any], catalog: Optional[str]=None, schema: Optional[str]=None) -> tuple[URL, dict[str, Any]]:
        if False:
            for i in range(10):
                print('nop')
        database = uri.database
        if schema and database:
            schema = parse.quote(schema, safe='')
            if '.' in database:
                database = database.split('.')[0] + '.' + schema
            else:
                database = 'default_catalog.' + schema
            uri = uri.set(database=database)
        return (uri, connect_args)

    @classmethod
    def get_schema_from_engine_params(cls, sqlalchemy_uri: URL, connect_args: dict[str, Any]) -> Optional[str]:
        if False:
            i = 10
            return i + 15
        '\n        Return the configured schema.\n\n        For StarRocks the SQLAlchemy URI looks like this:\n\n            starrocks://localhost:9030/catalog.schema\n\n        '
        database = sqlalchemy_uri.database.strip('/')
        if '.' not in database:
            return None
        return parse.unquote(database.split('.')[1])