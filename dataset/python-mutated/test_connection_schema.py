from __future__ import annotations
import re
import marshmallow
import pytest
from airflow.api_connexion.schemas.connection_schema import ConnectionCollection, connection_collection_item_schema, connection_collection_schema, connection_schema, connection_test_schema
from airflow.models import Connection
from airflow.utils.session import create_session, provide_session
from tests.test_utils.db import clear_db_connections
pytestmark = pytest.mark.db_test

class TestConnectionCollectionItemSchema:

    def setup_method(self) -> None:
        if False:
            i = 10
            return i + 15
        with create_session() as session:
            session.query(Connection).delete()

    def teardown_method(self) -> None:
        if False:
            return 10
        clear_db_connections()

    @provide_session
    def test_serialize(self, session):
        if False:
            i = 10
            return i + 15
        connection_model = Connection(conn_id='mysql_default', conn_type='mysql', host='mysql', login='login', schema='testschema', port=80)
        session.add(connection_model)
        session.commit()
        connection_model = session.query(Connection).first()
        deserialized_connection = connection_collection_item_schema.dump(connection_model)
        assert deserialized_connection == {'connection_id': 'mysql_default', 'conn_type': 'mysql', 'description': None, 'host': 'mysql', 'login': 'login', 'schema': 'testschema', 'port': 80}

    def test_deserialize(self):
        if False:
            for i in range(10):
                print('nop')
        connection_dump_1 = {'connection_id': 'mysql_default_1', 'conn_type': 'mysql', 'host': 'mysql', 'login': 'login', 'schema': 'testschema', 'port': 80}
        connection_dump_2 = {'connection_id': 'mysql_default_2', 'conn_type': 'postgres'}
        result_1 = connection_collection_item_schema.load(connection_dump_1)
        result_2 = connection_collection_item_schema.load(connection_dump_2)
        assert result_1 == {'conn_id': 'mysql_default_1', 'conn_type': 'mysql', 'host': 'mysql', 'login': 'login', 'schema': 'testschema', 'port': 80}
        assert result_2 == {'conn_id': 'mysql_default_2', 'conn_type': 'postgres'}

    def test_deserialize_required_fields(self):
        if False:
            print('Hello World!')
        connection_dump_1 = {'connection_id': 'mysql_default_2'}
        with pytest.raises(marshmallow.exceptions.ValidationError, match=re.escape("{'conn_type': ['Missing data for required field.']}")):
            connection_collection_item_schema.load(connection_dump_1)

class TestConnectionCollectionSchema:

    def setup_method(self) -> None:
        if False:
            while True:
                i = 10
        with create_session() as session:
            session.query(Connection).delete()

    def teardown_method(self) -> None:
        if False:
            return 10
        clear_db_connections()

    @provide_session
    def test_serialize(self, session):
        if False:
            print('Hello World!')
        connection_model_1 = Connection(conn_id='mysql_default_1', conn_type='test-type')
        connection_model_2 = Connection(conn_id='mysql_default_2', conn_type='test-type2')
        connections = [connection_model_1, connection_model_2]
        session.add_all(connections)
        session.commit()
        instance = ConnectionCollection(connections=connections, total_entries=2)
        deserialized_connections = connection_collection_schema.dump(instance)
        assert deserialized_connections == {'connections': [{'connection_id': 'mysql_default_1', 'conn_type': 'test-type', 'description': None, 'host': None, 'login': None, 'schema': None, 'port': None}, {'connection_id': 'mysql_default_2', 'conn_type': 'test-type2', 'description': None, 'host': None, 'login': None, 'schema': None, 'port': None}], 'total_entries': 2}

class TestConnectionSchema:

    def setup_method(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        with create_session() as session:
            session.query(Connection).delete()

    def teardown_method(self) -> None:
        if False:
            while True:
                i = 10
        clear_db_connections()

    @provide_session
    def test_serialize(self, session):
        if False:
            for i in range(10):
                print('nop')
        connection_model = Connection(conn_id='mysql_default', conn_type='mysql', host='mysql', login='login', schema='testschema', port=80, password='test-password', extra="{'key':'string'}")
        session.add(connection_model)
        session.commit()
        connection_model = session.query(Connection).first()
        deserialized_connection = connection_schema.dump(connection_model)
        assert deserialized_connection == {'connection_id': 'mysql_default', 'conn_type': 'mysql', 'description': None, 'host': 'mysql', 'login': 'login', 'schema': 'testschema', 'port': 80, 'extra': "{'key':'string'}"}

    def test_deserialize(self):
        if False:
            print('Hello World!')
        den = {'connection_id': 'mysql_default', 'conn_type': 'mysql', 'host': 'mysql', 'login': 'login', 'schema': 'testschema', 'port': 80, 'extra': "{'key':'string'}"}
        result = connection_schema.load(den)
        assert result == {'conn_id': 'mysql_default', 'conn_type': 'mysql', 'host': 'mysql', 'login': 'login', 'schema': 'testschema', 'port': 80, 'extra': "{'key':'string'}"}

class TestConnectionTestSchema:

    def test_response(self):
        if False:
            for i in range(10):
                print('nop')
        data = {'status': True, 'message': 'Connection tested successful'}
        result = connection_test_schema.load(data)
        assert result == {'status': True, 'message': 'Connection tested successful'}