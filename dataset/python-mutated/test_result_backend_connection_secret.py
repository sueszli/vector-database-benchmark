from __future__ import annotations
import base64
import jmespath
import pytest
from tests.charts.helm_template_generator import render_chart

class TestResultBackendConnectionSecret:
    """Tests result backend connection secret."""

    def _get_values_with_version(self, values, version):
        if False:
            return 10
        if version != 'default':
            values['airflowVersion'] = version
        return values

    def _assert_for_old_version(self, version, value, expected_value):
        if False:
            for i in range(10):
                print('nop')
        if version == '2.3.2':
            assert value == expected_value
        else:
            assert value is None
    non_chart_database_values = {'user': 'someuser', 'pass': 'somepass', 'host': 'somehost', 'protocol': 'postgresql', 'port': 7777, 'db': 'somedb', 'sslmode': 'allow'}

    def test_should_not_generate_a_document_if_using_existing_secret(self):
        if False:
            return 10
        docs = render_chart(values={'data': {'resultBackendSecretName': 'foo'}}, show_only=['templates/secrets/result-backend-connection-secret.yaml'])
        assert 0 == len(docs)

    @pytest.mark.parametrize('executor, expected_doc_count', [('CeleryExecutor', 1), ('CeleryKubernetesExecutor', 1), ('LocalExecutor', 0)])
    def test_should_a_document_be_generated_for_executor(self, executor, expected_doc_count):
        if False:
            for i in range(10):
                print('nop')
        docs = render_chart(values={'executor': executor, 'data': {'metadataConnection': {**self.non_chart_database_values}, 'resultBackendConnection': {**self.non_chart_database_values, 'user': 'anotheruser', 'pass': 'anotherpass'}}}, show_only=['templates/secrets/result-backend-connection-secret.yaml'])
        assert expected_doc_count == len(docs)

    def _get_connection(self, values: dict) -> str | None:
        if False:
            i = 10
            return i + 15
        docs = render_chart(values=values, show_only=['templates/secrets/result-backend-connection-secret.yaml'])
        if len(docs) == 0:
            return None
        encoded_connection = jmespath.search('data.connection', docs[0])
        return base64.b64decode(encoded_connection).decode()

    @pytest.mark.parametrize('version', ['2.3.2', '2.4.0', 'default'])
    def test_default_connection_old_version(self, version):
        if False:
            return 10
        connection = self._get_connection(self._get_values_with_version(version=version, values={}))
        self._assert_for_old_version(version, value=connection, expected_value='db+postgresql://postgres:postgres@release-name-postgresql:5432/postgres?sslmode=disable')

    @pytest.mark.parametrize('version', ['2.3.2', '2.4.0', 'default'])
    def test_should_default_to_custom_metadata_db_connection_with_pgbouncer_overrides(self, version):
        if False:
            while True:
                i = 10
        values = {'pgbouncer': {'enabled': True}, 'data': {'metadataConnection': {**self.non_chart_database_values}}}
        connection = self._get_connection(self._get_values_with_version(values=values, version=version))
        self._assert_for_old_version(version, value=connection, expected_value='db+postgresql://someuser:somepass@release-name-pgbouncer:6543/release-name-result-backend?sslmode=allow')

    @pytest.mark.parametrize('version', ['2.3.2', '2.4.0', 'default'])
    def test_should_set_pgbouncer_overrides_when_enabled(self, version):
        if False:
            for i in range(10):
                print('nop')
        values = {'pgbouncer': {'enabled': True}}
        connection = self._get_connection(self._get_values_with_version(values=values, version=version))
        self._assert_for_old_version(version, value=connection, expected_value='db+postgresql://postgres:postgres@release-name-pgbouncer:6543/release-name-result-backend?sslmode=disable')

    def test_should_set_pgbouncer_overrides_with_non_chart_database_when_enabled(self):
        if False:
            while True:
                i = 10
        values = {'pgbouncer': {'enabled': True}, 'data': {'resultBackendConnection': {**self.non_chart_database_values}}}
        connection = self._get_connection(values)
        assert 'db+postgresql://someuser:somepass@release-name-pgbouncer:6543/release-name-result-backend?sslmode=allow' == connection

    @pytest.mark.parametrize('version', ['2.3.2', '2.4.0', 'default'])
    def test_should_default_to_custom_metadata_db_connection_in_old_version(self, version):
        if False:
            print('Hello World!')
        values = {'data': {'metadataConnection': {**self.non_chart_database_values}}}
        connection = self._get_connection(self._get_values_with_version(values=values, version=version))
        self._assert_for_old_version(version, value=connection, expected_value='db+postgresql://someuser:somepass@somehost:7777/somedb?sslmode=allow')

    def test_should_correctly_use_non_chart_database(self):
        if False:
            while True:
                i = 10
        values = {'data': {'resultBackendConnection': {**self.non_chart_database_values}}}
        connection = self._get_connection(values)
        assert 'db+postgresql://someuser:somepass@somehost:7777/somedb?sslmode=allow' == connection

    def test_should_support_non_postgres_db(self):
        if False:
            for i in range(10):
                print('nop')
        values = {'data': {'resultBackendConnection': {**self.non_chart_database_values, 'protocol': 'mysql'}}}
        connection = self._get_connection(values)
        assert 'db+mysql://someuser:somepass@somehost:7777/somedb' == connection

    def test_should_correctly_use_non_chart_database_when_both_db_are_external(self):
        if False:
            return 10
        values = {'data': {'metadataConnection': {**self.non_chart_database_values}, 'resultBackendConnection': {**self.non_chart_database_values, 'user': 'anotheruser', 'pass': 'anotherpass'}}}
        connection = self._get_connection(values)
        assert 'db+postgresql://anotheruser:anotherpass@somehost:7777/somedb?sslmode=allow' == connection

    def test_should_correctly_handle_password_with_special_characters(self):
        if False:
            for i in range(10):
                print('nop')
        values = {'data': {'resultBackendConnection': {**self.non_chart_database_values, 'user': 'username@123123', 'pass': 'password@!@#$^&*()'}}}
        connection = self._get_connection(values)
        assert 'db+postgresql://username%40123123:password%40%21%40%23$%5E&%2A%28%29@somehost:7777/somedb?sslmode=allow' == connection