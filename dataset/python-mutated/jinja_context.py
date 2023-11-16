"""Defines the templating context for SQL Lab"""
import json
import re
from datetime import datetime
from functools import lru_cache, partial
from typing import Any, Callable, cast, Optional, TYPE_CHECKING, TypedDict, Union
import dateutil
from flask import current_app, g, has_request_context, request
from flask_babel import gettext as _
from jinja2 import DebugUndefined
from jinja2.sandbox import SandboxedEnvironment
from sqlalchemy.engine.interfaces import Dialect
from sqlalchemy.sql.expression import bindparam
from sqlalchemy.types import String
from superset.constants import LRU_CACHE_MAX_SIZE
from superset.datasets.commands.exceptions import DatasetNotFoundError
from superset.exceptions import SupersetTemplateException
from superset.extensions import feature_flag_manager
from superset.utils.core import convert_legacy_filters_into_adhoc, get_user_id, merge_extra_filters
if TYPE_CHECKING:
    from superset.connectors.sqla.models import SqlaTable
    from superset.models.core import Database
    from superset.models.sql_lab import Query
NONE_TYPE = type(None).__name__
ALLOWED_TYPES = (NONE_TYPE, 'bool', 'str', 'unicode', 'int', 'long', 'float', 'list', 'dict', 'tuple', 'set')
COLLECTION_TYPES = ('list', 'dict', 'tuple', 'set')

@lru_cache(maxsize=LRU_CACHE_MAX_SIZE)
def context_addons() -> dict[str, Any]:
    if False:
        for i in range(10):
            print('nop')
    return current_app.config.get('JINJA_CONTEXT_ADDONS', {})

class Filter(TypedDict):
    op: str
    col: str
    val: Union[None, Any, list[Any]]

class ExtraCache:
    """
    Dummy class that exposes a method used to store additional values used in
    calculation of query object cache keys.
    """
    regex = re.compile('\\{\\{.*(current_user_id\\(.*\\)|current_username\\(.*\\)|cache_key_wrapper\\(.*\\)|url_param\\(.*\\)).*\\}\\}')

    def __init__(self, extra_cache_keys: Optional[list[Any]]=None, applied_filters: Optional[list[str]]=None, removed_filters: Optional[list[str]]=None, dialect: Optional[Dialect]=None):
        if False:
            return 10
        self.extra_cache_keys = extra_cache_keys
        self.applied_filters = applied_filters if applied_filters is not None else []
        self.removed_filters = removed_filters if removed_filters is not None else []
        self.dialect = dialect

    def current_user_id(self, add_to_cache_keys: bool=True) -> Optional[int]:
        if False:
            return 10
        '\n        Return the user ID of the user who is currently logged in.\n\n        :param add_to_cache_keys: Whether the value should be included in the cache key\n        :returns: The user ID\n        '
        if hasattr(g, 'user') and g.user:
            id_ = get_user_id()
            if add_to_cache_keys:
                self.cache_key_wrapper(id_)
            return id_
        return None

    def current_username(self, add_to_cache_keys: bool=True) -> Optional[str]:
        if False:
            return 10
        '\n        Return the username of the user who is currently logged in.\n\n        :param add_to_cache_keys: Whether the value should be included in the cache key\n        :returns: The username\n        '
        if g.user and hasattr(g.user, 'username'):
            if add_to_cache_keys:
                self.cache_key_wrapper(g.user.username)
            return g.user.username
        return None

    def cache_key_wrapper(self, key: Any) -> Any:
        if False:
            while True:
                i = 10
        '\n        Adds values to a list that is added to the query object used for calculating a\n        cache key.\n\n        This is needed if the following applies:\n            - Caching is enabled\n            - The query is dynamically generated using a jinja template\n            - A `JINJA_CONTEXT_ADDONS` or similar is used as a filter in the query\n\n        :param key: Any value that should be considered when calculating the cache key\n        :return: the original value ``key`` passed to the function\n        '
        if self.extra_cache_keys is not None:
            self.extra_cache_keys.append(key)
        return key

    def url_param(self, param: str, default: Optional[str]=None, add_to_cache_keys: bool=True, escape_result: bool=True) -> Optional[str]:
        if False:
            return 10
        '\n        Read a url or post parameter and use it in your SQL Lab query.\n\n        When in SQL Lab, it\'s possible to add arbitrary URL "query string" parameters,\n        and use those in your SQL code. For instance you can alter your url and add\n        `?foo=bar`, as in `{domain}/sqllab?foo=bar`. Then if your query is\n        something like SELECT * FROM foo = \'{{ url_param(\'foo\') }}\', it will be parsed\n        at runtime and replaced by the value in the URL.\n\n        As you create a visualization form this SQL Lab query, you can pass parameters\n        in the explore view as well as from the dashboard, and it should carry through\n        to your queries.\n\n        Default values for URL parameters can be defined in chart metadata by adding the\n        key-value pair `url_params: {\'foo\': \'bar\'}`\n\n        :param param: the parameter to lookup\n        :param default: the value to return in the absence of the parameter\n        :param add_to_cache_keys: Whether the value should be included in the cache key\n        :param escape_result: Should special characters in the result be escaped\n        :returns: The URL parameters\n        '
        from superset.views.utils import get_form_data
        if has_request_context() and request.args.get(param):
            return request.args.get(param, default)
        (form_data, _) = get_form_data()
        url_params = form_data.get('url_params') or {}
        result = url_params.get(param, default)
        if result and escape_result and self.dialect:
            result = String().literal_processor(dialect=self.dialect)(value=result)[1:-1]
        if add_to_cache_keys:
            self.cache_key_wrapper(result)
        return result

    def filter_values(self, column: str, default: Optional[str]=None, remove_filter: bool=False) -> list[Any]:
        if False:
            print('Hello World!')
        'Gets a values for a particular filter as a list\n\n        This is useful if:\n            - you want to use a filter component to filter a query where the name of\n             filter component column doesn\'t match the one in the select statement\n            - you want to have the ability for filter inside the main query for speed\n            purposes\n\n        Usage example::\n\n            SELECT action, count(*) as times\n            FROM logs\n            WHERE\n                action in ({{ "\'" + "\',\'".join(filter_values(\'action_type\')) + "\'" }})\n            GROUP BY action\n\n        :param column: column/filter name to lookup\n        :param default: default value to return if there\'s no matching columns\n        :param remove_filter: When set to true, mark the filter as processed,\n            removing it from the outer query. Useful when a filter should\n            only apply to the inner query\n        :return: returns a list of filter values\n        '
        return_val: list[Any] = []
        filters = self.get_filters(column, remove_filter)
        for flt in filters:
            val = flt.get('val')
            if isinstance(val, list):
                return_val.extend(val)
            elif val:
                return_val.append(val)
        if not return_val and default:
            return_val = [default]
        return return_val

    def get_filters(self, column: str, remove_filter: bool=False) -> list[Filter]:
        if False:
            for i in range(10):
                print('nop')
        'Get the filters applied to the given column. In addition\n           to returning values like the filter_values function\n           the get_filters function returns the operator specified in the explorer UI.\n\n        This is useful if:\n            - you want to handle more than the IN operator in your SQL clause\n            - you want to handle generating custom SQL conditions for a filter\n            - you want to have the ability for filter inside the main query for speed\n            purposes\n\n        Usage example::\n\n\n            WITH RECURSIVE\n                superiors(employee_id, manager_id, full_name, level, lineage) AS (\n                SELECT\n                    employee_id,\n                    manager_id,\n                    full_name,\n                1 as level,\n                employee_id as lineage\n                FROM\n                    employees\n                WHERE\n                1=1\n                {# Render a blank line #}\n                {%- for filter in get_filters(\'full_name\', remove_filter=True) -%}\n                {%- if filter.get(\'op\') == \'IN\' -%}\n                    AND\n                    full_name IN ( {{ "\'" + "\', \'".join(filter.get(\'val\')) + "\'" }} )\n                {%- endif -%}\n                {%- if filter.get(\'op\') == \'LIKE\' -%}\n                    AND\n                    full_name LIKE {{ "\'" + filter.get(\'val\') + "\'" }}\n                {%- endif -%}\n                {%- endfor -%}\n                UNION ALL\n                    SELECT\n                        e.employee_id,\n                        e.manager_id,\n                        e.full_name,\n                s.level + 1 as level,\n                s.lineage\n                    FROM\n                        employees e,\n                    superiors s\n                    WHERE s.manager_id = e.employee_id\n            )\n\n\n            SELECT\n                employee_id, manager_id, full_name, level, lineage\n            FROM\n                superiors\n            order by lineage, level\n\n        :param column: column/filter name to lookup\n        :param remove_filter: When set to true, mark the filter as processed,\n            removing it from the outer query. Useful when a filter should\n            only apply to the inner query\n        :return: returns a list of filters\n        '
        from superset.utils.core import FilterOperator
        from superset.views.utils import get_form_data
        (form_data, _) = get_form_data()
        convert_legacy_filters_into_adhoc(form_data)
        merge_extra_filters(form_data)
        filters: list[Filter] = []
        for flt in form_data.get('adhoc_filters', []):
            val: Union[Any, list[Any]] = flt.get('comparator')
            op: str = flt['operator'].upper() if flt.get('operator') else None
            if flt.get('expressionType') == 'SIMPLE' and flt.get('clause') == 'WHERE' and (flt.get('subject') == column) and val:
                if remove_filter:
                    if column not in self.removed_filters:
                        self.removed_filters.append(column)
                if column not in self.applied_filters:
                    self.applied_filters.append(column)
                if op in (FilterOperator.IN.value, FilterOperator.NOT_IN.value) and (not isinstance(val, list)):
                    val = [val]
                filters.append({'op': op, 'col': column, 'val': val})
        return filters

def safe_proxy(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    if False:
        print('Hello World!')
    return_value = func(*args, **kwargs)
    value_type = type(return_value).__name__
    if value_type not in ALLOWED_TYPES:
        raise SupersetTemplateException(_('Unsafe return type for function %(func)s: %(value_type)s', func=func.__name__, value_type=value_type))
    if value_type in COLLECTION_TYPES:
        try:
            return_value = json.loads(json.dumps(return_value))
        except TypeError as ex:
            raise SupersetTemplateException(_('Unsupported return value for method %(name)s', name=func.__name__)) from ex
    return return_value

def validate_context_types(context: dict[str, Any]) -> dict[str, Any]:
    if False:
        for i in range(10):
            print('nop')
    for key in context:
        arg_type = type(context[key]).__name__
        if arg_type not in ALLOWED_TYPES and key not in context_addons():
            if arg_type == 'partial' and context[key].func.__name__ == 'safe_proxy':
                continue
            raise SupersetTemplateException(_('Unsafe template value for key %(key)s: %(value_type)s', key=key, value_type=arg_type))
        if arg_type in COLLECTION_TYPES:
            try:
                context[key] = json.loads(json.dumps(context[key]))
            except TypeError as ex:
                raise SupersetTemplateException(_('Unsupported template value for key %(key)s', key=key)) from ex
    return context

def validate_template_context(engine: Optional[str], context: dict[str, Any]) -> dict[str, Any]:
    if False:
        return 10
    if engine and engine in context:
        engine_context = validate_context_types(context.pop(engine))
        valid_context = validate_context_types(context)
        valid_context[engine] = engine_context
        return valid_context
    return validate_context_types(context)

class WhereInMacro:

    def __init__(self, dialect: Dialect):
        if False:
            i = 10
            return i + 15
        self.dialect = dialect

    def __call__(self, values: list[Any], mark: Optional[str]=None) -> str:
        if False:
            return 10
        '\n        Given a list of values, build a parenthesis list suitable for an IN expression.\n\n            >>> from sqlalchemy.dialects import mysql\n            >>> where_in = WhereInMacro(dialect=mysql.dialect())\n            >>> where_in([1, "Joe\'s", 3])\n            (1, \'Joe\'\'s\', 3)\n\n        '
        binds = [bindparam(f'value_{i}', value) for (i, value) in enumerate(values)]
        string_representations = [str(bind.compile(dialect=self.dialect, compile_kwargs={'literal_binds': True})) for bind in binds]
        joined_values = ', '.join(string_representations)
        result = f'({joined_values})'
        if mark:
            result += '\n-- WARNING: the `mark` parameter was removed from the `where_in` macro for security reasons\n'
        return result

class BaseTemplateProcessor:
    """
    Base class for database-specific jinja context
    """
    engine: Optional[str] = None

    def __init__(self, database: 'Database', query: Optional['Query']=None, table: Optional['SqlaTable']=None, extra_cache_keys: Optional[list[Any]]=None, removed_filters: Optional[list[str]]=None, applied_filters: Optional[list[str]]=None, **kwargs: Any) -> None:
        if False:
            print('Hello World!')
        self._database = database
        self._query = query
        self._schema = None
        if query and query.schema:
            self._schema = query.schema
        elif table:
            self._schema = table.schema
        self._extra_cache_keys = extra_cache_keys
        self._applied_filters = applied_filters
        self._removed_filters = removed_filters
        self._context: dict[str, Any] = {}
        self._env = SandboxedEnvironment(undefined=DebugUndefined)
        self.set_context(**kwargs)
        self._env.filters['where_in'] = WhereInMacro(database.get_dialect())

    def set_context(self, **kwargs: Any) -> None:
        if False:
            print('Hello World!')
        self._context.update(kwargs)
        self._context.update(context_addons())

    def process_template(self, sql: str, **kwargs: Any) -> str:
        if False:
            return 10
        'Processes a sql template\n\n        >>> sql = "SELECT \'{{ datetime(2017, 1, 1).isoformat() }}\'"\n        >>> process_template(sql)\n        "SELECT \'2017-01-01T00:00:00\'"\n        '
        template = self._env.from_string(sql)
        kwargs.update(self._context)
        context = validate_template_context(self.engine, kwargs)
        return template.render(context)

class JinjaTemplateProcessor(BaseTemplateProcessor):

    def _parse_datetime(self, dttm: str) -> Optional[datetime]:
        if False:
            return 10
        '\n        Try to parse a datetime and default to None in the worst case.\n\n        Since this may have been rendered by different engines, the datetime may\n        vary slightly in format. We try to make it consistent, and if all else\n        fails, just return None.\n        '
        try:
            return dateutil.parser.parse(dttm)
        except dateutil.parser.ParserError:
            return None

    def set_context(self, **kwargs: Any) -> None:
        if False:
            while True:
                i = 10
        super().set_context(**kwargs)
        extra_cache = ExtraCache(extra_cache_keys=self._extra_cache_keys, applied_filters=self._applied_filters, removed_filters=self._removed_filters, dialect=self._database.get_dialect())
        from_dttm = self._parse_datetime(dttm) if (dttm := self._context.get('from_dttm')) else None
        to_dttm = self._parse_datetime(dttm) if (dttm := self._context.get('to_dttm')) else None
        dataset_macro_with_context = partial(dataset_macro, from_dttm=from_dttm, to_dttm=to_dttm)
        self._context.update({'url_param': partial(safe_proxy, extra_cache.url_param), 'current_user_id': partial(safe_proxy, extra_cache.current_user_id), 'current_username': partial(safe_proxy, extra_cache.current_username), 'cache_key_wrapper': partial(safe_proxy, extra_cache.cache_key_wrapper), 'filter_values': partial(safe_proxy, extra_cache.filter_values), 'get_filters': partial(safe_proxy, extra_cache.get_filters), 'dataset': partial(safe_proxy, dataset_macro_with_context)})

class NoOpTemplateProcessor(BaseTemplateProcessor):

    def process_template(self, sql: str, **kwargs: Any) -> str:
        if False:
            print('Hello World!')
        '\n        Makes processing a template a noop\n        '
        return sql

class PrestoTemplateProcessor(JinjaTemplateProcessor):
    """Presto Jinja context

    The methods described here are namespaced under ``presto`` in the
    jinja context as in ``SELECT '{{ presto.some_macro_call() }}'``
    """
    engine = 'presto'

    def set_context(self, **kwargs: Any) -> None:
        if False:
            while True:
                i = 10
        super().set_context(**kwargs)
        self._context[self.engine] = {'first_latest_partition': partial(safe_proxy, self.first_latest_partition), 'latest_partitions': partial(safe_proxy, self.latest_partitions), 'latest_sub_partition': partial(safe_proxy, self.latest_sub_partition), 'latest_partition': partial(safe_proxy, self.latest_partition)}

    @staticmethod
    def _schema_table(table_name: str, schema: Optional[str]) -> tuple[str, Optional[str]]:
        if False:
            for i in range(10):
                print('nop')
        if '.' in table_name:
            (schema, table_name) = table_name.split('.')
        return (table_name, schema)

    def first_latest_partition(self, table_name: str) -> Optional[str]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Gets the first value in the array of all latest partitions\n\n        :param table_name: table name in the format `schema.table`\n        :return: the first (or only) value in the latest partition array\n        :raises IndexError: If no partition exists\n        '
        latest_partitions = self.latest_partitions(table_name)
        return latest_partitions[0] if latest_partitions else None

    def latest_partitions(self, table_name: str) -> Optional[list[str]]:
        if False:
            return 10
        '\n        Gets the array of all latest partitions\n\n        :param table_name: table name in the format `schema.table`\n        :return: the latest partition array\n        '
        from superset.db_engine_specs.presto import PrestoEngineSpec
        (table_name, schema) = self._schema_table(table_name, self._schema)
        return cast(PrestoEngineSpec, self._database.db_engine_spec).latest_partition(table_name, schema, self._database)[1]

    def latest_sub_partition(self, table_name: str, **kwargs: Any) -> Any:
        if False:
            for i in range(10):
                print('nop')
        (table_name, schema) = self._schema_table(table_name, self._schema)
        from superset.db_engine_specs.presto import PrestoEngineSpec
        return cast(PrestoEngineSpec, self._database.db_engine_spec).latest_sub_partition(table_name=table_name, schema=schema, database=self._database, **kwargs)
    latest_partition = first_latest_partition

class HiveTemplateProcessor(PrestoTemplateProcessor):
    engine = 'hive'

class TrinoTemplateProcessor(PrestoTemplateProcessor):
    engine = 'trino'

    def process_template(self, sql: str, **kwargs: Any) -> str:
        if False:
            print('Hello World!')
        template = self._env.from_string(sql)
        kwargs.update(self._context)
        context = validate_template_context(self.engine, kwargs)
        context['presto'] = context['trino']
        return template.render(context)
DEFAULT_PROCESSORS = {'presto': PrestoTemplateProcessor, 'hive': HiveTemplateProcessor, 'trino': TrinoTemplateProcessor}

@lru_cache(maxsize=LRU_CACHE_MAX_SIZE)
def get_template_processors() -> dict[str, Any]:
    if False:
        for i in range(10):
            print('nop')
    processors = current_app.config.get('CUSTOM_TEMPLATE_PROCESSORS', {})
    for (engine, processor) in DEFAULT_PROCESSORS.items():
        if engine not in processors:
            processors[engine] = processor
    return processors

def get_template_processor(database: 'Database', table: Optional['SqlaTable']=None, query: Optional['Query']=None, **kwargs: Any) -> BaseTemplateProcessor:
    if False:
        for i in range(10):
            print('nop')
    if feature_flag_manager.is_feature_enabled('ENABLE_TEMPLATE_PROCESSING'):
        template_processor = get_template_processors().get(database.backend, JinjaTemplateProcessor)
    else:
        template_processor = NoOpTemplateProcessor
    return template_processor(database=database, table=table, query=query, **kwargs)

def dataset_macro(dataset_id: int, include_metrics: bool=False, columns: Optional[list[str]]=None, from_dttm: Optional[datetime]=None, to_dttm: Optional[datetime]=None) -> str:
    if False:
        print('Hello World!')
    '\n    Given a dataset ID, return the SQL that represents it.\n\n    The generated SQL includes all columns (including computed) by default. Optionally\n    the user can also request metrics to be included, and columns to group by.\n\n    The from_dttm and to_dttm parameters are filled in from filter values in explore\n    views, and we take them to make those properties available to jinja templates in\n    the underlying dataset.\n    '
    from superset.daos.dataset import DatasetDAO
    dataset = DatasetDAO.find_by_id(dataset_id)
    if not dataset:
        raise DatasetNotFoundError(f'Dataset {dataset_id} not found!')
    columns = columns or [column.column_name for column in dataset.columns]
    metrics = [metric.metric_name for metric in dataset.metrics]
    query_obj = {'is_timeseries': False, 'filter': [], 'metrics': metrics if include_metrics else None, 'columns': columns, 'from_dttm': from_dttm, 'to_dttm': to_dttm}
    sqla_query = dataset.get_query_str_extended(query_obj, mutate=False)
    sql = sqla_query.sql
    return f'(\n{sql}\n) AS dataset_{dataset_id}'