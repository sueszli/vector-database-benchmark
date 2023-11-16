import contextlib
import re
import threading
from re import Pattern
from typing import Any, Callable, List, NamedTuple, Optional
from flask_babel import gettext as __
from sqlalchemy.engine.reflection import Inspector
from sqlalchemy.orm import Session
with contextlib.suppress(ImportError, RuntimeError):
    import geojson
    import pyocient
    from shapely import wkt
    from superset import app
    superset_log_level = app.config['LOG_LEVEL']
    pyocient.logger.setLevel(superset_log_level)
from superset.constants import TimeGrain
from superset.db_engine_specs.base import BaseEngineSpec
from superset.errors import SupersetErrorType
from superset.models.core import Database
from superset.models.sql_lab import Query
CONNECTION_INVALID_USERNAME_REGEX = re.compile("The referenced user does not exist \\(User '(?P<username>.*?)' not found\\)")
CONNECTION_INVALID_PASSWORD_REGEX = re.compile('The userid/password combination was not valid \\(Incorrect password for user\\)')
CONNECTION_INVALID_HOSTNAME_REGEX = re.compile('Unable to connect to (?P<host>.*?):(?P<port>.*?)')
CONNECTION_UNKNOWN_DATABASE_REGEX = re.compile("No database named '(?P<database>.*?)' exists")
CONNECTION_INVALID_PORT_ERROR = re.compile('Port out of range 0-65535')
INVALID_CONNECTION_STRING_REGEX = re.compile('An invalid connection string attribute was specified \\(failed to decrypt cipher text\\)')
SYNTAX_ERROR_REGEX = re.compile("There is a syntax error in your statement \\((?P<qualifier>.*?) input '(?P<input>.*?)' expecting (?P<expected>.*?)\\)")
TABLE_DOES_NOT_EXIST_REGEX = re.compile("The referenced table or view '(?P<table>.*?)' does not exist")
COLUMN_DOES_NOT_EXIST_REGEX = re.compile("The reference to column '(?P<column>.*?)' is not valid")

def _to_hex(data: bytes) -> str:
    if False:
        print('Hello World!')
    '\n    Converts the bytes object into a string of hexadecimal digits.\n\n    :param data: the bytes object\n    :returns: string of hexadecimal digits representing the bytes\n    '
    return data.hex()

def _wkt_to_geo_json(geo_as_wkt: str) -> Any:
    if False:
        for i in range(10):
            print('nop')
    '\n    Converts pyocient geometry objects to their geoJSON representation.\n\n    :param geo_as_wkt: the GIS object in WKT format\n    :returns: the geoJSON encoding of `geo`\n    '
    geo = wkt.loads(geo_as_wkt)
    return geojson.Feature(geometry=geo, properties={})

def _point_list_to_wkt(points) -> str:
    if False:
        while True:
            i = 10
    '\n    Converts the list of pyocient._STPoint elements to a WKT LineString.\n\n    :param points: the list of pyocient._STPoint objects\n    :returns: WKT LineString\n    '
    coords = [f'{p.long} {p.lat}' for p in points]
    return f"LINESTRING({', '.join(coords)})"

def _point_to_geo_json(point) -> Any:
    if False:
        i = 10
        return i + 15
    '\n    Converts the pyocient._STPolygon object to the geoJSON format\n\n    :param point: the pyocient._STPoint instance\n    :returns: the geoJSON encoding of this point\n    '
    wkt_point = str(point)
    return _wkt_to_geo_json(wkt_point)

def _linestring_to_geo_json(linestring) -> Any:
    if False:
        for i in range(10):
            print('nop')
    '\n    Converts the pyocient._STLinestring object to a GIS format\n    compatible with the Superset visualization toolkit (powered\n    by Deck.gl).\n\n    :param linestring: the pyocient._STLinestring instance\n    :returns: the geoJSON of this linestring\n    '
    if len(linestring.points) == 1:
        point = linestring.points[0]
        return _point_to_geo_json(point)
    wkt_linestring = str(linestring)
    return _wkt_to_geo_json(wkt_linestring)

def _polygon_to_geo_json(polygon) -> Any:
    if False:
        while True:
            i = 10
    '\n    Converts the pyocient._STPolygon object to a GIS format\n    compatible with the Superset visualization toolkit (powered\n    by Deck.gl).\n\n    :param polygon: the pyocient._STPolygon instance\n    :returns: the geoJSON encoding of this polygon\n    '
    if len(polygon.exterior) > 0 and len(polygon.holes) == 0:
        if len(polygon.exterior) == 1:
            point = polygon.exterior[0]
            return _point_to_geo_json(point)
        if polygon.exterior[0] != polygon.exterior[-1]:
            wkt_linestring = _point_list_to_wkt(polygon.exterior)
            return _wkt_to_geo_json(wkt_linestring)
    wkt_polygon = str(polygon)
    return _wkt_to_geo_json(wkt_polygon)
SanitizeFunc = Callable[[Any], Any]

class PlacedSanitizeFunc(NamedTuple):
    column_index: int
    sanitize_func: SanitizeFunc
try:
    from pyocient import TypeCodes
    _sanitized_ocient_type_codes: dict[int, SanitizeFunc] = {TypeCodes.BINARY: _to_hex, TypeCodes.ST_POINT: _point_to_geo_json, TypeCodes.IP: str, TypeCodes.IPV4: str, TypeCodes.ST_LINESTRING: _linestring_to_geo_json, TypeCodes.ST_POLYGON: _polygon_to_geo_json}
except ImportError as e:
    _sanitized_ocient_type_codes = {}

def _find_columns_to_sanitize(cursor: Any) -> list[PlacedSanitizeFunc]:
    if False:
        while True:
            i = 10
    '\n    Cleans the column value for consumption by Superset.\n\n    :param cursor: the result set cursor\n    :returns: the list of tuples consisting of the column index and sanitization function\n    '
    return [PlacedSanitizeFunc(i, _sanitized_ocient_type_codes[cursor.description[i][1]]) for i in range(len(cursor.description)) if cursor.description[i][1] in _sanitized_ocient_type_codes]

class OcientEngineSpec(BaseEngineSpec):
    engine = 'ocient'
    engine_name = 'Ocient'
    force_column_alias_quotes = True
    max_column_name_length = 30
    allows_cte_in_subquery = False
    cte_alias = 'cte__'
    query_id_mapping: dict[str, str] = {}
    query_id_mapping_lock = threading.Lock()
    custom_errors: dict[Pattern[str], tuple[str, SupersetErrorType, dict[str, Any]]] = {CONNECTION_INVALID_USERNAME_REGEX: (__('The username "%(username)s" does not exist.'), SupersetErrorType.CONNECTION_INVALID_USERNAME_ERROR, {}), CONNECTION_INVALID_PASSWORD_REGEX: (__('The user/password combination is not valid (Incorrect password for user).'), SupersetErrorType.CONNECTION_INVALID_PASSWORD_ERROR, {}), CONNECTION_UNKNOWN_DATABASE_REGEX: (__('Could not connect to database: "%(database)s"'), SupersetErrorType.CONNECTION_UNKNOWN_DATABASE_ERROR, {}), CONNECTION_INVALID_HOSTNAME_REGEX: (__('Could not resolve hostname: "%(host)s".'), SupersetErrorType.CONNECTION_INVALID_HOSTNAME_ERROR, {}), CONNECTION_INVALID_PORT_ERROR: (__('Port out of range 0-65535'), SupersetErrorType.CONNECTION_INVALID_PORT_ERROR, {}), INVALID_CONNECTION_STRING_REGEX: (__("Invalid Connection String: Expecting String of the form 'ocient://user:pass@host:port/database'."), SupersetErrorType.GENERIC_DB_ENGINE_ERROR, {}), SYNTAX_ERROR_REGEX: (__('Syntax Error: %(qualifier)s input "%(input)s" expecting "%(expected)s'), SupersetErrorType.SYNTAX_ERROR, {}), TABLE_DOES_NOT_EXIST_REGEX: (__('Table or View "%(table)s" does not exist.'), SupersetErrorType.TABLE_DOES_NOT_EXIST_ERROR, {}), COLUMN_DOES_NOT_EXIST_REGEX: (__('Invalid reference to column: "%(column)s"'), SupersetErrorType.COLUMN_DOES_NOT_EXIST_ERROR, {})}
    _time_grain_expressions = {None: '{col}', TimeGrain.SECOND: "ROUND({col}, 'SECOND')", TimeGrain.MINUTE: "ROUND({col}, 'MINUTE')", TimeGrain.HOUR: "ROUND({col}, 'HOUR')", TimeGrain.DAY: "ROUND({col}, 'DAY')", TimeGrain.WEEK: "ROUND({col}, 'WEEK')", TimeGrain.MONTH: "ROUND({col}, 'MONTH')", TimeGrain.QUARTER_YEAR: "ROUND({col}, 'QUARTER')", TimeGrain.YEAR: "ROUND({col}, 'YEAR')"}

    @classmethod
    def get_table_names(cls, database: Database, inspector: Inspector, schema: Optional[str]) -> set[str]:
        if False:
            return 10
        return inspector.get_table_names(schema)

    @classmethod
    def fetch_data(cls, cursor: Any, limit: Optional[int]=None) -> list[tuple[Any, ...]]:
        if False:
            i = 10
            return i + 15
        try:
            rows: list[tuple[Any, ...]] = super().fetch_data(cursor, limit)
        except Exception as exception:
            with OcientEngineSpec.query_id_mapping_lock:
                del OcientEngineSpec.query_id_mapping[getattr(cursor, 'superset_query_id')]
            raise exception
        if len(rows) > 0 and type(rows[0]).__name__ == 'Row':
            columns_to_sanitize: list[PlacedSanitizeFunc] = _find_columns_to_sanitize(cursor)
            if columns_to_sanitize:

                def identity(x: Any) -> Any:
                    if False:
                        for i in range(10):
                            print('nop')
                    return x
                sanitization_functions: list[SanitizeFunc] = [identity for _ in range(len(cursor.description))]
                for info in columns_to_sanitize:
                    sanitization_functions[info.column_index] = info.sanitize_func
                rows = [tuple((sanitize_func(val) for (sanitize_func, val) in zip(sanitization_functions, row))) for row in rows]
        return rows

    @classmethod
    def epoch_to_dttm(cls) -> str:
        if False:
            return 10
        return "DATEADD(S, {col}, '1970-01-01')"

    @classmethod
    def epoch_ms_to_dttm(cls) -> str:
        if False:
            print('Hello World!')
        return "DATEADD(MS, {col}, '1970-01-01')"

    @classmethod
    def get_cancel_query_id(cls, cursor: Any, query: Query) -> Optional[str]:
        if False:
            return 10
        return 'DUMMY_VALUE'

    @classmethod
    def handle_cursor(cls, cursor: Any, query: Query, session: Session) -> None:
        if False:
            i = 10
            return i + 15
        with OcientEngineSpec.query_id_mapping_lock:
            OcientEngineSpec.query_id_mapping[query.id] = cursor.query_id
        setattr(cursor, 'superset_query_id', query.id)
        return super().handle_cursor(cursor, query, session)

    @classmethod
    def cancel_query(cls, cursor: Any, query: Query, cancel_query_id: str) -> bool:
        if False:
            for i in range(10):
                print('nop')
        with OcientEngineSpec.query_id_mapping_lock:
            if query.id in OcientEngineSpec.query_id_mapping:
                cursor.execute(f'CANCEL {OcientEngineSpec.query_id_mapping[query.id]}')
                del OcientEngineSpec.query_id_mapping[query.id]
                return True
            return False