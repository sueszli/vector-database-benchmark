"""Unit tests for Sql Lab"""
import unittest
from unittest.mock import MagicMock, patch
from pyhive.exc import DatabaseError
from superset.sql_validators.postgres import PostgreSQLValidator
from superset.sql_validators.presto_db import PrestoDBSQLValidator, PrestoSQLValidationError
from superset.utils.database import get_example_database
from .base_tests import SupersetTestCase

class TestPrestoValidator(SupersetTestCase):
    """Testing for the prestodb sql validator"""

    def setUp(self):
        if False:
            print('Hello World!')
        self.validator = PrestoDBSQLValidator
        self.database = MagicMock()
        self.database_engine = self.database.get_sqla_engine_with_context.return_value.__enter__.return_value
        self.database_conn = self.database_engine.raw_connection.return_value
        self.database_cursor = self.database_conn.cursor.return_value
        self.database_cursor.poll.return_value = None

    def tearDown(self):
        if False:
            return 10
        self.logout()
    PRESTO_ERROR_TEMPLATE = {'errorLocation': {'lineNumber': 10, 'columnNumber': 20}, 'message': "your query isn't how I like it"}

    @patch('superset.utils.core.g')
    def test_validator_success(self, flask_g):
        if False:
            while True:
                i = 10
        flask_g.user.username = 'nobody'
        sql = 'SELECT 1 FROM default.notarealtable'
        schema = 'default'
        errors = self.validator.validate(sql, schema, self.database)
        self.assertEqual([], errors)

    @patch('superset.utils.core.g')
    def test_validator_db_error(self, flask_g):
        if False:
            i = 10
            return i + 15
        flask_g.user.username = 'nobody'
        sql = 'SELECT 1 FROM default.notarealtable'
        schema = 'default'
        fetch_fn = self.database.db_engine_spec.fetch_data
        fetch_fn.side_effect = DatabaseError('dummy db error')
        with self.assertRaises(PrestoSQLValidationError):
            self.validator.validate(sql, schema, self.database)

    @patch('superset.utils.core.g')
    def test_validator_unexpected_error(self, flask_g):
        if False:
            return 10
        flask_g.user.username = 'nobody'
        sql = 'SELECT 1 FROM default.notarealtable'
        schema = 'default'
        fetch_fn = self.database.db_engine_spec.fetch_data
        fetch_fn.side_effect = Exception('a mysterious failure')
        with self.assertRaises(Exception):
            self.validator.validate(sql, schema, self.database)

    @patch('superset.utils.core.g')
    def test_validator_query_error(self, flask_g):
        if False:
            while True:
                i = 10
        flask_g.user.username = 'nobody'
        sql = 'SELECT 1 FROM default.notarealtable'
        schema = 'default'
        fetch_fn = self.database.db_engine_spec.fetch_data
        fetch_fn.side_effect = DatabaseError(self.PRESTO_ERROR_TEMPLATE)
        errors = self.validator.validate(sql, schema, self.database)
        self.assertEqual(1, len(errors))

class TestPostgreSQLValidator(SupersetTestCase):

    def test_valid_syntax(self):
        if False:
            for i in range(10):
                print('nop')
        if get_example_database().backend != 'postgresql':
            return
        mock_database = MagicMock()
        annotations = PostgreSQLValidator.validate(sql='SELECT 1, "col" FROM "table"', schema='', database=mock_database)
        assert annotations == []

    def test_invalid_syntax(self):
        if False:
            print('Hello World!')
        if get_example_database().backend != 'postgresql':
            return
        mock_database = MagicMock()
        annotations = PostgreSQLValidator.validate(sql='SELECT 1, "col"\nFROOM "table"', schema='', database=mock_database)
        assert len(annotations) == 1
        annotation = annotations[0]
        assert annotation.line_number == 2
        assert annotation.start_column is None
        assert annotation.end_column is None
        assert annotation.message == 'ERROR: syntax error at or near """'
if __name__ == '__main__':
    unittest.main()