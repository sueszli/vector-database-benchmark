import dataclasses
import logging
import uuid
from contextlib import closing
from datetime import datetime
from sys import getsizeof
from typing import Any, cast, Optional, Union
import backoff
import msgpack
import simplejson as json
from celery import Task
from celery.exceptions import SoftTimeLimitExceeded
from flask_babel import gettext as __
from sqlalchemy.orm import Session
from superset import app, db, is_feature_enabled, results_backend, results_backend_use_msgpack, security_manager
from superset.common.db_query_status import QueryStatus
from superset.constants import QUERY_CANCEL_KEY, QUERY_EARLY_CANCEL_KEY
from superset.dataframe import df_to_records
from superset.db_engine_specs import BaseEngineSpec
from superset.errors import ErrorLevel, SupersetError, SupersetErrorType
from superset.exceptions import SupersetErrorException, SupersetErrorsException
from superset.extensions import celery_app
from superset.models.core import Database
from superset.models.sql_lab import Query
from superset.result_set import SupersetResultSet
from superset.sql_parse import CtasMethod, insert_rls_as_subquery, insert_rls_in_predicate, ParsedQuery
from superset.sqllab.limiting_factor import LimitingFactor
from superset.sqllab.utils import write_ipc_buffer
from superset.utils.celery import session_scope
from superset.utils.core import json_iso_dttm_ser, override_user, QuerySource, zlib_compress
from superset.utils.dates import now_as_float
from superset.utils.decorators import stats_timing
config = app.config
stats_logger = config['STATS_LOGGER']
SQLLAB_TIMEOUT = config['SQLLAB_ASYNC_TIME_LIMIT_SEC']
SQLLAB_HARD_TIMEOUT = SQLLAB_TIMEOUT + 60
SQL_MAX_ROW = config['SQL_MAX_ROW']
SQLLAB_CTAS_NO_LIMIT = config['SQLLAB_CTAS_NO_LIMIT']
SQL_QUERY_MUTATOR = config['SQL_QUERY_MUTATOR']
log_query = config['QUERY_LOGGER']
logger = logging.getLogger(__name__)

class SqlLabException(Exception):
    pass

class SqlLabSecurityException(SqlLabException):
    pass

class SqlLabQueryStoppedException(SqlLabException):
    pass

def handle_query_error(ex: Exception, query: Query, session: Session, payload: Optional[dict[str, Any]]=None, prefix_message: str='') -> dict[str, Any]:
    if False:
        while True:
            i = 10
    'Local method handling error while processing the SQL'
    payload = payload or {}
    msg = f'{prefix_message} {str(ex)}'.strip()
    query.error_message = msg
    query.tmp_table_name = None
    query.status = QueryStatus.FAILED
    if not query.end_time:
        query.end_time = now_as_float()
    if isinstance(ex, SupersetErrorException):
        errors = [ex.error]
    elif isinstance(ex, SupersetErrorsException):
        errors = ex.errors
    else:
        errors = query.database.db_engine_spec.extract_errors(str(ex))
    errors_payload = [dataclasses.asdict(error) for error in errors]
    if errors:
        query.set_extra_json_key('errors', errors_payload)
    session.commit()
    payload.update({'status': query.status, 'error': msg, 'errors': errors_payload})
    if (troubleshooting_link := config['TROUBLESHOOTING_LINK']):
        payload['link'] = troubleshooting_link
    return payload

def get_query_backoff_handler(details: dict[Any, Any]) -> None:
    if False:
        return 10
    query_id = details['kwargs']['query_id']
    logger.error('Query with id `%s` could not be retrieved', str(query_id), exc_info=True)
    stats_logger.incr(f"error_attempting_orm_query_{details['tries'] - 1}")
    logger.error('Query %s: Sleeping for a sec before retrying...', str(query_id), exc_info=True)

def get_query_giveup_handler(_: Any) -> None:
    if False:
        for i in range(10):
            print('nop')
    stats_logger.incr('error_failed_at_getting_orm_query')

@backoff.on_exception(backoff.constant, SqlLabException, interval=1, on_backoff=get_query_backoff_handler, on_giveup=get_query_giveup_handler, max_tries=5)
def get_query(query_id: int, session: Session) -> Query:
    if False:
        while True:
            i = 10
    'attempts to get the query and retry if it cannot'
    try:
        return session.query(Query).filter_by(id=query_id).one()
    except Exception as ex:
        raise SqlLabException('Failed at getting query') from ex

@celery_app.task(name='sql_lab.get_sql_results', bind=True, time_limit=SQLLAB_HARD_TIMEOUT, soft_time_limit=SQLLAB_TIMEOUT)
def get_sql_results(ctask: Task, query_id: int, rendered_query: str, return_results: bool=True, store_results: bool=False, username: Optional[str]=None, start_time: Optional[float]=None, expand_data: bool=False, log_params: Optional[dict[str, Any]]=None) -> Optional[dict[str, Any]]:
    if False:
        i = 10
        return i + 15
    'Executes the sql query returns the results.'
    with session_scope(not ctask.request.called_directly) as session:
        with override_user(security_manager.find_user(username)):
            try:
                return execute_sql_statements(query_id, rendered_query, return_results, store_results, session=session, start_time=start_time, expand_data=expand_data, log_params=log_params)
            except Exception as ex:
                logger.debug('Query %d: %s', query_id, ex)
                stats_logger.incr('error_sqllab_unhandled')
                query = get_query(query_id, session)
                return handle_query_error(ex, query, session)

def execute_sql_statement(sql_statement: str, query: Query, session: Session, cursor: Any, log_params: Optional[dict[str, Any]], apply_ctas: bool=False) -> SupersetResultSet:
    if False:
        for i in range(10):
            print('nop')
    'Executes a single SQL statement'
    database: Database = query.database
    db_engine_spec = database.db_engine_spec
    parsed_query = ParsedQuery(sql_statement)
    if is_feature_enabled('RLS_IN_SQLLAB'):
        insert_rls = insert_rls_as_subquery if database.db_engine_spec.allows_subqueries and database.db_engine_spec.allows_alias_in_select else insert_rls_in_predicate
        parsed_query = ParsedQuery(str(insert_rls(parsed_query._parsed[0], database.id, query.schema)))
    sql = parsed_query.stripped()
    increased_limit = None if query.limit is None else query.limit + 1
    if not db_engine_spec.is_readonly_query(parsed_query) and (not database.allow_dml):
        raise SupersetErrorException(SupersetError(message=__('Only SELECT statements are allowed against this database.'), error_type=SupersetErrorType.DML_NOT_ALLOWED_ERROR, level=ErrorLevel.ERROR))
    if apply_ctas:
        if not query.tmp_table_name:
            start_dttm = datetime.fromtimestamp(query.start_time)
            query.tmp_table_name = f"tmp_{query.user_id}_table_{start_dttm.strftime('%Y_%m_%d_%H_%M_%S')}"
        sql = parsed_query.as_create_table(query.tmp_table_name, schema_name=query.tmp_schema_name, method=query.ctas_method)
        query.select_as_cta_used = True
    if db_engine_spec.is_select_query(parsed_query) and (not (query.select_as_cta_used and SQLLAB_CTAS_NO_LIMIT)):
        if SQL_MAX_ROW and (not query.limit or query.limit > SQL_MAX_ROW):
            query.limit = SQL_MAX_ROW
        sql = apply_limit_if_exists(database, increased_limit, query, sql)
    sql = SQL_QUERY_MUTATOR(sql, security_manager=security_manager, database=database)
    try:
        query.executed_sql = sql
        if log_query:
            log_query(query.database.sqlalchemy_uri, query.executed_sql, query.schema, __name__, security_manager, log_params)
        session.commit()
        with stats_timing('sqllab.query.time_executing_query', stats_logger):
            db_engine_spec.execute_with_cursor(cursor, sql, query, session)
        with stats_timing('sqllab.query.time_fetching_results', stats_logger):
            logger.debug('Query %d: Fetching data for query object: %s', query.id, str(query.to_dict()))
            data = db_engine_spec.fetch_data(cursor, increased_limit)
            if query.limit is None or len(data) <= query.limit:
                query.limiting_factor = LimitingFactor.NOT_LIMITED
            else:
                data = data[:-1]
    except SoftTimeLimitExceeded as ex:
        query.status = QueryStatus.TIMED_OUT
        logger.warning('Query %d: Time limit exceeded', query.id)
        logger.debug('Query %d: %s', query.id, ex)
        raise SupersetErrorException(SupersetError(message=__('The query was killed after %(sqllab_timeout)s seconds. It might be too complex, or the database might be under heavy load.', sqllab_timeout=SQLLAB_TIMEOUT), error_type=SupersetErrorType.SQLLAB_TIMEOUT_ERROR, level=ErrorLevel.ERROR)) from ex
    except Exception as ex:
        session.refresh(query)
        if query.status == QueryStatus.STOPPED:
            raise SqlLabQueryStoppedException() from ex
        logger.debug('Query %d: %s', query.id, ex)
        raise SqlLabException(db_engine_spec.extract_error_message(ex)) from ex
    logger.debug('Query %d: Fetching cursor description', query.id)
    cursor_description = cursor.description
    return SupersetResultSet(data, cursor_description, db_engine_spec)

def apply_limit_if_exists(database: Database, increased_limit: Optional[int], query: Query, sql: str) -> str:
    if False:
        print('Hello World!')
    if query.limit and increased_limit:
        sql = database.apply_limit_to_sql(sql, increased_limit, force=True)
    return sql

def _serialize_payload(payload: dict[Any, Any], use_msgpack: Optional[bool]=False) -> Union[bytes, str]:
    if False:
        print('Hello World!')
    logger.debug('Serializing to msgpack: %r', use_msgpack)
    if use_msgpack:
        return msgpack.dumps(payload, default=json_iso_dttm_ser, use_bin_type=True)
    return json.dumps(payload, default=json_iso_dttm_ser, ignore_nan=True)

def _serialize_and_expand_data(result_set: SupersetResultSet, db_engine_spec: BaseEngineSpec, use_msgpack: Optional[bool]=False, expand_data: bool=False) -> tuple[Union[bytes, str], list[Any], list[Any], list[Any]]:
    if False:
        return 10
    selected_columns = result_set.columns
    all_columns: list[Any]
    expanded_columns: list[Any]
    if use_msgpack:
        with stats_timing('sqllab.query.results_backend_pa_serialization', stats_logger):
            data = write_ipc_buffer(result_set.pa_table).to_pybytes()
        (all_columns, expanded_columns) = (selected_columns, [])
    else:
        df = result_set.to_pandas_df()
        data = df_to_records(df) or []
        if expand_data:
            (all_columns, data, expanded_columns) = db_engine_spec.expand_data(selected_columns, data)
        else:
            all_columns = selected_columns
            expanded_columns = []
    return (data, selected_columns, all_columns, expanded_columns)

def execute_sql_statements(query_id: int, rendered_query: str, return_results: bool, store_results: bool, session: Session, start_time: Optional[float], expand_data: bool, log_params: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    if False:
        for i in range(10):
            print('nop')
    'Executes the sql query returns the results.'
    if store_results and start_time:
        stats_logger.timing('sqllab.query.time_pending', now_as_float() - start_time)
    query = get_query(query_id, session)
    payload: dict[str, Any] = {'query_id': query_id}
    database = query.database
    db_engine_spec = database.db_engine_spec
    db_engine_spec.patch()
    if database.allow_run_async and (not results_backend):
        raise SupersetErrorException(SupersetError(message=__('Results backend is not configured.'), error_type=SupersetErrorType.RESULTS_BACKEND_NOT_CONFIGURED_ERROR, level=ErrorLevel.ERROR))
    parsed_query = ParsedQuery(rendered_query, strip_comments=True)
    if not db_engine_spec.run_multiple_statements_as_one:
        statements = parsed_query.get_statements()
        logger.info('Query %s: Executing %i statement(s)', str(query_id), len(statements))
    else:
        statements = [rendered_query]
        logger.info('Query %s: Executing query as a single statement', str(query_id))
    logger.info("Query %s: Set query to 'running'", str(query_id))
    query.status = QueryStatus.RUNNING
    query.start_running_time = now_as_float()
    session.commit()
    if query.select_as_cta and query.ctas_method == CtasMethod.TABLE and (not parsed_query.is_valid_ctas()):
        raise SupersetErrorException(SupersetError(message=__('CTAS (create table as select) can only be run with a query where the last statement is a SELECT. Please make sure your query has a SELECT as its last statement. Then, try running your query again.'), error_type=SupersetErrorType.INVALID_CTAS_QUERY_ERROR, level=ErrorLevel.ERROR))
    if query.select_as_cta and query.ctas_method == CtasMethod.VIEW and (not parsed_query.is_valid_cvas()):
        raise SupersetErrorException(SupersetError(message=__('CVAS (create view as select) can only be run with a query with a single SELECT statement. Please make sure your query has only a SELECT statement. Then, try running your query again.'), error_type=SupersetErrorType.INVALID_CVAS_QUERY_ERROR, level=ErrorLevel.ERROR))
    with database.get_raw_connection(query.schema, source=QuerySource.SQL_LAB) as conn:
        cursor = conn.cursor()
        cancel_query_id = db_engine_spec.get_cancel_query_id(cursor, query)
        if cancel_query_id is not None:
            query.set_extra_json_key(QUERY_CANCEL_KEY, cancel_query_id)
            session.commit()
        statement_count = len(statements)
        for (i, statement) in enumerate(statements):
            session.refresh(query)
            if query.status == QueryStatus.STOPPED:
                payload.update({'status': query.status})
                return payload
            apply_ctas = query.select_as_cta and (query.ctas_method == CtasMethod.VIEW or (query.ctas_method == CtasMethod.TABLE and i == len(statements) - 1))
            msg = __('Running statement %(statement_num)s out of %(statement_count)s', statement_num=i + 1, statement_count=statement_count)
            logger.info('Query %s: %s', str(query_id), msg)
            query.set_extra_json_key('progress', msg)
            session.commit()
            try:
                result_set = execute_sql_statement(statement, query, session, cursor, log_params, apply_ctas)
            except SqlLabQueryStoppedException:
                payload.update({'status': QueryStatus.STOPPED})
                return payload
            except Exception as ex:
                msg = str(ex)
                prefix_message = __('Statement %(statement_num)s out of %(statement_count)s', statement_num=i + 1, statement_count=statement_count) if statement_count > 1 else ''
                payload = handle_query_error(ex, query, session, payload, prefix_message)
                return payload
        should_commit = not db_engine_spec.is_select_query(parsed_query) or apply_ctas
        if should_commit:
            conn.commit()
    query.rows = result_set.size
    query.progress = 100
    query.set_extra_json_key('progress', None)
    query.set_extra_json_key('columns', result_set.columns)
    if query.select_as_cta:
        query.select_sql = database.select_star(query.tmp_table_name, schema=query.tmp_schema_name, limit=query.limit, show_cols=False, latest_partition=False)
    query.end_time = now_as_float()
    use_arrow_data = store_results and cast(bool, results_backend_use_msgpack)
    (data, selected_columns, all_columns, expanded_columns) = _serialize_and_expand_data(result_set, db_engine_spec, use_arrow_data, expand_data)
    payload.update({'status': QueryStatus.SUCCESS, 'data': data, 'columns': all_columns, 'selected_columns': selected_columns, 'expanded_columns': expanded_columns, 'query': query.to_dict()})
    payload['query']['state'] = QueryStatus.SUCCESS
    if store_results and results_backend:
        key = str(uuid.uuid4())
        payload['query']['resultsKey'] = key
        logger.info('Query %s: Storing results in results backend, key: %s', str(query_id), key)
        with stats_timing('sqllab.query.results_backend_write', stats_logger):
            with stats_timing('sqllab.query.results_backend_write_serialization', stats_logger):
                serialized_payload = _serialize_payload(payload, cast(bool, results_backend_use_msgpack))
            cache_timeout = database.cache_timeout
            if cache_timeout is None:
                cache_timeout = config['CACHE_DEFAULT_TIMEOUT']
            compressed = zlib_compress(serialized_payload)
            logger.debug('*** serialized payload size: %i', getsizeof(serialized_payload))
            logger.debug('*** compressed payload size: %i', getsizeof(compressed))
            results_backend.set(key, compressed, cache_timeout)
        query.results_key = key
    query.status = QueryStatus.SUCCESS
    session.commit()
    if return_results:
        if use_arrow_data:
            (data, selected_columns, all_columns, expanded_columns) = _serialize_and_expand_data(result_set, db_engine_spec, False, expand_data)
            payload.update({'data': data, 'columns': all_columns, 'selected_columns': selected_columns, 'expanded_columns': expanded_columns})
        return payload
    return None

def cancel_query(query: Query) -> bool:
    if False:
        return 10
    '\n    Cancel a running query.\n\n    Note some engines implicitly handle the cancelation of a query and thus no explicit\n    action is required.\n\n    :param query: Query to cancel\n    :return: True if query cancelled successfully, False otherwise\n    '
    if query.database.db_engine_spec.has_implicit_cancel():
        return True
    query.database.db_engine_spec.prepare_cancel_query(query, db.session)
    if query.extra.get(QUERY_EARLY_CANCEL_KEY):
        return True
    cancel_query_id = query.extra.get(QUERY_CANCEL_KEY)
    if cancel_query_id is None:
        return False
    with query.database.get_sqla_engine_with_context(query.schema, source=QuerySource.SQL_LAB) as engine:
        with closing(engine.raw_connection()) as conn:
            with closing(conn.cursor()) as cursor:
                return query.database.db_engine_spec.cancel_query(cursor, query, cancel_query_id)