import logging
import time
from contextlib import closing
from typing import Any, Optional
from superset import app, security_manager
from superset.models.core import Database
from superset.sql_parse import ParsedQuery
from superset.sql_validators.base import BaseSQLValidator, SQLValidationAnnotation
from superset.utils.core import QuerySource
MAX_ERROR_ROWS = 10
config = app.config
logger = logging.getLogger(__name__)

class PrestoSQLValidationError(Exception):
    """Error in the process of asking Presto to validate SQL querytext"""

class PrestoDBSQLValidator(BaseSQLValidator):
    """Validate SQL queries using Presto's built-in EXPLAIN subtype"""
    name = 'PrestoDBSQLValidator'

    @classmethod
    def validate_statement(cls, statement: str, database: Database, cursor: Any) -> Optional[SQLValidationAnnotation]:
        if False:
            for i in range(10):
                print('nop')
        db_engine_spec = database.db_engine_spec
        parsed_query = ParsedQuery(statement)
        sql = parsed_query.stripped()
        if (sql_query_mutator := config['SQL_QUERY_MUTATOR']):
            sql = sql_query_mutator(sql, security_manager=security_manager, database=database)
        sql = f'EXPLAIN (TYPE VALIDATE) {sql}'
        from pyhive.exc import DatabaseError
        try:
            db_engine_spec.execute(cursor, sql)
            polled = cursor.poll()
            while polled:
                logger.info('polling presto for validation progress')
                stats = polled.get('stats', {})
                if stats:
                    state = stats.get('state')
                    if state == 'FINISHED':
                        break
                time.sleep(0.2)
                polled = cursor.poll()
            db_engine_spec.fetch_data(cursor, MAX_ERROR_ROWS)
            return None
        except DatabaseError as db_error:
            if db_error.args and isinstance(db_error.args[0], str):
                raise PrestoSQLValidationError(db_error.args[0]) from db_error
            if not db_error.args or not isinstance(db_error.args[0], dict):
                raise PrestoSQLValidationError('The pyhive presto client returned an unhandled database error.') from db_error
            error_args: dict[str, Any] = db_error.args[0]
            if 'message' not in error_args:
                raise PrestoSQLValidationError('The pyhive presto client did not report an error message') from db_error
            if 'errorLocation' not in error_args:
                message = error_args['message'] + '\n(Error location unknown)'
                return SQLValidationAnnotation(message=message, line_number=1, start_column=1, end_column=1)
            message = error_args['message']
            err_loc = error_args['errorLocation']
            line_number = err_loc.get('lineNumber', None)
            start_column = err_loc.get('columnNumber', None)
            end_column = err_loc.get('columnNumber', None)
            return SQLValidationAnnotation(message=message, line_number=line_number, start_column=start_column, end_column=end_column)
        except Exception as ex:
            logger.exception('Unexpected error running validation query: %s', str(ex))
            raise ex

    @classmethod
    def validate(cls, sql: str, schema: Optional[str], database: Database) -> list[SQLValidationAnnotation]:
        if False:
            return 10
        '\n        Presto supports query-validation queries by running them with a\n        prepended explain.\n\n        For example, "SELECT 1 FROM default.mytable" becomes "EXPLAIN (TYPE\n        VALIDATE) SELECT 1 FROM default.mytable.\n        '
        parsed_query = ParsedQuery(sql)
        statements = parsed_query.get_statements()
        logger.info('Validating %i statement(s)', len(statements))
        with database.get_sqla_engine_with_context(schema, source=QuerySource.SQL_LAB) as engine:
            annotations: list[SQLValidationAnnotation] = []
            with closing(engine.raw_connection()) as conn:
                cursor = conn.cursor()
                for statement in parsed_query.get_statements():
                    annotation = cls.validate_statement(statement, database, cursor)
                    if annotation:
                        annotations.append(annotation)
            logger.debug('Validation found %i error(s)', len(annotations))
        return annotations