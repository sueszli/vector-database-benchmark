from __future__ import annotations
import base64
import jmespath
from tests.charts.helm_template_generator import render_chart

class TestMetadataConnectionSecret:
    """Tests metadata connection secret."""
    non_chart_database_values = {'user': 'someuser', 'pass': 'somepass', 'host': 'somehost', 'port': 7777, 'db': 'somedb'}

    def test_should_not_generate_a_document_if_using_existing_secret(self):
        if False:
            i = 10
            return i + 15
        docs = render_chart(values={'data': {'metadataSecretName': 'foo'}}, show_only=['templates/secrets/metadata-connection-secret.yaml'])
        assert 0 == len(docs)

    def _get_connection(self, values: dict) -> str:
        if False:
            i = 10
            return i + 15
        docs = render_chart(values=values, show_only=['templates/secrets/metadata-connection-secret.yaml'])
        encoded_connection = jmespath.search('data.connection', docs[0])
        return base64.b64decode(encoded_connection).decode()

    def test_default_connection(self):
        if False:
            while True:
                i = 10
        connection = self._get_connection({})
        assert 'postgresql://postgres:postgres@release-name-postgresql.default:5432/postgres?sslmode=disable' == connection

    def test_should_set_pgbouncer_overrides_when_enabled(self):
        if False:
            while True:
                i = 10
        values = {'pgbouncer': {'enabled': True}}
        connection = self._get_connection(values)
        assert 'postgresql://postgres:postgres@release-name-pgbouncer.default:6543/release-name-metadata?sslmode=disable' == connection

    def test_should_set_pgbouncer_overrides_with_non_chart_database_when_enabled(self):
        if False:
            while True:
                i = 10
        values = {'pgbouncer': {'enabled': True}, 'data': {'metadataConnection': {**self.non_chart_database_values}}}
        connection = self._get_connection(values)
        assert 'postgresql://someuser:somepass@release-name-pgbouncer.default:6543/release-name-metadata?sslmode=disable' == connection

    def test_should_correctly_use_non_chart_database(self):
        if False:
            while True:
                i = 10
        values = {'data': {'metadataConnection': {**self.non_chart_database_values, 'sslmode': 'require'}}}
        connection = self._get_connection(values)
        assert 'postgresql://someuser:somepass@somehost:7777/somedb?sslmode=require' == connection

    def test_should_support_non_postgres_db(self):
        if False:
            print('Hello World!')
        values = {'data': {'metadataConnection': {**self.non_chart_database_values, 'protocol': 'mysql'}}}
        connection = self._get_connection(values)
        assert 'mysql://someuser:somepass@somehost:7777/somedb' == connection

    def test_should_correctly_handle_password_with_special_characters(self):
        if False:
            return 10
        values = {'data': {'metadataConnection': {**self.non_chart_database_values, 'user': 'username@123123', 'pass': 'password@!@#$^&*()'}}}
        connection = self._get_connection(values)
        assert 'postgresql://username%40123123:password%40%21%40%23$%5E&%2A%28%29@somehost:7777/somedb?sslmode=disable' == connection