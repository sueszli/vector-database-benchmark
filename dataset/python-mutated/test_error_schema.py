from __future__ import annotations
import pytest
from airflow.api_connexion.schemas.error_schema import ImportErrorCollection, import_error_collection_schema, import_error_schema
from airflow.models.errors import ImportError
from airflow.utils import timezone
from airflow.utils.session import provide_session
from tests.test_utils.db import clear_db_import_errors
pytestmark = pytest.mark.db_test

class TestErrorSchemaBase:

    def setup_method(self) -> None:
        if False:
            return 10
        clear_db_import_errors()
        self.timestamp = '2020-06-10T12:02:44'

    def teardown_method(self) -> None:
        if False:
            print('Hello World!')
        clear_db_import_errors()

class TestErrorSchema(TestErrorSchemaBase):

    @provide_session
    def test_serialize(self, session):
        if False:
            i = 10
            return i + 15
        import_error = ImportError(filename='lorem.py', stacktrace='Lorem Ipsum', timestamp=timezone.parse(self.timestamp, timezone='UTC'))
        session.add(import_error)
        session.commit()
        serialized_data = import_error_schema.dump(import_error)
        serialized_data['import_error_id'] = 1
        assert {'filename': 'lorem.py', 'import_error_id': 1, 'stack_trace': 'Lorem Ipsum', 'timestamp': '2020-06-10T12:02:44+00:00'} == serialized_data

class TestErrorCollectionSchema(TestErrorSchemaBase):

    @provide_session
    def test_serialize(self, session):
        if False:
            i = 10
            return i + 15
        import_error = [ImportError(filename='Lorem_ipsum.py', stacktrace='Lorem ipsum', timestamp=timezone.parse(self.timestamp, timezone='UTC')) for i in range(2)]
        session.add_all(import_error)
        session.commit()
        query = session.query(ImportError)
        query_list = query.all()
        serialized_data = import_error_collection_schema.dump(ImportErrorCollection(import_errors=query_list, total_entries=2))
        serialized_data['import_errors'][0]['import_error_id'] = 1
        serialized_data['import_errors'][1]['import_error_id'] = 2
        assert {'import_errors': [{'filename': 'Lorem_ipsum.py', 'import_error_id': 1, 'stack_trace': 'Lorem ipsum', 'timestamp': '2020-06-10T12:02:44+00:00'}, {'filename': 'Lorem_ipsum.py', 'import_error_id': 2, 'stack_trace': 'Lorem ipsum', 'timestamp': '2020-06-10T12:02:44+00:00'}], 'total_entries': 2} == serialized_data