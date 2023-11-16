import logging
import unittest
from apache_beam.io.debezium import DriverClassName
from apache_beam.io.debezium import ReadFromDebezium
from apache_beam.options.pipeline_options import StandardOptions
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.util import assert_that
from apache_beam.testing.util import equal_to
try:
    from testcontainers.postgres import PostgresContainer
except ImportError:
    PostgresContainer = None
NUM_RECORDS = 1

@unittest.skipIf(PostgresContainer is None, 'testcontainers package is not installed')
@unittest.skipIf(TestPipeline().get_pipeline_options().view_as(StandardOptions).runner is None, 'Do not run this test on precommit suites.')
class CrossLanguageDebeziumIOTest(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.username = 'debezium'
        self.password = 'dbz'
        self.database = 'inventory'
        self.start_db_container(retries=1)
        self.host = self.db.get_container_host_ip()
        self.port = self.db.get_exposed_port(5432)
        self.connector_class = DriverClassName.POSTGRESQL
        self.connection_properties = ['database.dbname=inventory', 'database.server.name=dbserver1', 'database.include.list=inventory', 'include.schema.changes=false']

    def tearDown(self):
        if False:
            return 10
        try:
            self.db.stop()
        except:
            logging.error('Could not stop the DB container.')

    def test_xlang_debezium_read(self):
        if False:
            return 10
        expected_response = [{'metadata': {'connector': 'postgresql', 'version': '1.3.1.Final', 'name': 'dbserver1', 'database': 'inventory', 'schema': 'inventory', 'table': 'customers'}, 'before': None, 'after': {'fields': {'last_name': 'Thomas', 'id': 1001, 'first_name': 'Sally', 'email': 'sally.thomas@acme.com'}}}]
        with TestPipeline() as p:
            p.not_use_test_runner_api = True
            results = p | 'Read from debezium' >> ReadFromDebezium(username=self.username, password=self.password, host=self.host, port=self.port, max_number_of_records=NUM_RECORDS, connector_class=self.connector_class, connection_properties=self.connection_properties)
            assert_that(results, equal_to(expected_response))

    def start_db_container(self, retries):
        if False:
            return 10
        for i in range(retries):
            try:
                self.db = PostgresContainer('debezium/example-postgres:latest', user=self.username, password=self.password, dbname=self.database)
                self.db.start()
                break
            except Exception as e:
                if i == retries - 1:
                    logging.error('Unable to initialize DB container.')
                    raise e
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()