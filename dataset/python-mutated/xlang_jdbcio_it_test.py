import datetime
import logging
import time
import typing
import unittest
from decimal import Decimal
from typing import Callable
from typing import Union
from parameterized import parameterized
import apache_beam as beam
from apache_beam import coders
from apache_beam.io.jdbc import ReadFromJdbc
from apache_beam.io.jdbc import WriteToJdbc
from apache_beam.options.pipeline_options import StandardOptions
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.util import assert_that
from apache_beam.testing.util import equal_to
from apache_beam.typehints.schemas import LogicalType
from apache_beam.typehints.schemas import MillisInstant
from apache_beam.utils.timestamp import Timestamp
try:
    import sqlalchemy
except ImportError:
    sqlalchemy = None
try:
    from testcontainers.postgres import PostgresContainer
    from testcontainers.mysql import MySqlContainer
except ImportError:
    PostgresContainer = None
ROW_COUNT = 10
JdbcTestRow = typing.NamedTuple('JdbcTestRow', [('f_id', int), ('f_float', float), ('f_char', str), ('f_varchar', str), ('f_bytes', bytes), ('f_varbytes', bytes), ('f_timestamp', Timestamp), ('f_decimal', Decimal), ('f_date', datetime.date), ('f_time', datetime.time)])
coders.registry.register_coder(JdbcTestRow, coders.RowCoder)

@unittest.skipIf(sqlalchemy is None, 'sql alchemy package is not installed.')
@unittest.skipIf(PostgresContainer is None, 'testcontainers package is not installed')
@unittest.skipIf(TestPipeline().get_pipeline_options().view_as(StandardOptions).runner is None, 'Do not run this test on precommit suites.')
class CrossLanguageJdbcIOTest(unittest.TestCase):
    DbData = typing.NamedTuple('DbData', [('container_fn', typing.Any), ('classpath', typing.List[str]), ('db_string', str), ('connector', str)])
    DB_CONTAINER_CLASSPATH_STRING = {'postgres': DbData(lambda : PostgresContainer('postgres:12.3'), None, 'postgresql', 'org.postgresql.Driver'), 'mysql': DbData(lambda : MySqlContainer(), ['mysql:mysql-connector-java:8.0.28'], 'mysql', 'com.mysql.cj.jdbc.Driver')}

    def _setUpTestCase(self, container_init: Callable[[], Union[PostgresContainer, MySqlContainer]], db_string: str, driver: str):
        if False:
            for i in range(10):
                print('nop')
        self.start_db_container(retries=3, container_init=container_init)
        self.engine = sqlalchemy.create_engine(self.db.get_connection_url())
        self.username = 'test'
        self.password = 'test'
        self.host = self.db.get_container_host_ip()
        self.port = self.db.get_exposed_port(self.db.port_to_expose)
        self.database_name = 'test'
        self.driver_class_name = driver
        self.jdbc_url = 'jdbc:{}://{}:{}/{}'.format(db_string, self.host, self.port, self.database_name)

    def tearDown(self):
        if False:
            print('Hello World!')
        try:
            self.db.stop()
        except:
            logging.error('Could not stop the postgreSQL container.')

    @parameterized.expand(['postgres', 'mysql'])
    def test_xlang_jdbc_write_read(self, database):
        if False:
            i = 10
            return i + 15
        (container_init, classpath, db_string, driver) = CrossLanguageJdbcIOTest.DB_CONTAINER_CLASSPATH_STRING[database]
        self._setUpTestCase(container_init, db_string, driver)
        table_name = 'jdbc_external_test'
        if database == 'postgres':
            binary_type = ('BYTEA', 'BYTEA')
        else:
            binary_type = ('BINARY(10)', 'VARBINARY(10)')
        self.engine.execute('CREATE TABLE IF NOT EXISTS {}'.format(table_name) + '(f_id INTEGER, ' + 'f_float DOUBLE PRECISION, ' + 'f_char CHAR(10), ' + 'f_varchar VARCHAR(10), ' + f'f_bytes {binary_type[0]}, ' + f'f_varbytes {binary_type[1]}, ' + 'f_timestamp TIMESTAMP(3), ' + 'f_decimal DECIMAL(10, 2), ' + 'f_date DATE, ' + 'f_time TIME(3))')
        inserted_rows = [JdbcTestRow(i, i + 0.1, f'Test{i}', f'Test{i}', f'Test{i}'.encode(), f'Test{i}'.encode(), Timestamp.of(seconds=round(time.time(), 3)), Decimal(f'{i - 1}.23'), datetime.date(1969 + i, i % 12 + 1, i % 31 + 1), datetime.time(i % 24, i % 60, i % 60, i * 1000 % 1000000)) for i in range(ROW_COUNT)]
        expected_row = []
        for row in inserted_rows:
            f_char = row.f_char + ' ' * (10 - len(row.f_char))
            if database != 'postgres':
                f_bytes = row.f_bytes + b'\x00' * (10 - len(row.f_bytes))
            else:
                f_bytes = row.f_bytes
            expected_row.append(JdbcTestRow(row.f_id, row.f_float, f_char, row.f_varchar, f_bytes, row.f_bytes, row.f_timestamp, row.f_decimal, row.f_date, row.f_time))
        with TestPipeline() as p:
            p.not_use_test_runner_api = True
            _ = p | beam.Create(inserted_rows).with_output_types(JdbcTestRow) | 'Write to jdbc' >> WriteToJdbc(table_name=table_name, driver_class_name=self.driver_class_name, jdbc_url=self.jdbc_url, username=self.username, password=self.password, classpath=classpath)
        LogicalType.register_logical_type(MillisInstant)
        with TestPipeline() as p:
            p.not_use_test_runner_api = True
            result = p | 'Read from jdbc' >> ReadFromJdbc(table_name=table_name, driver_class_name=self.driver_class_name, jdbc_url=self.jdbc_url, username=self.username, password=self.password, classpath=classpath)
            assert_that(result, equal_to(expected_row))
        with TestPipeline() as p:
            p.not_use_test_runner_api = True
            result = p | 'Partitioned read from jdbc' >> ReadFromJdbc(table_name=table_name, partition_column='f_id', partitions=3, driver_class_name=self.driver_class_name, jdbc_url=self.jdbc_url, username=self.username, password=self.password, classpath=classpath)
            assert_that(result, equal_to(expected_row))

    def start_db_container(self, retries, container_init):
        if False:
            i = 10
            return i + 15
        for i in range(retries):
            try:
                self.db = container_init()
                self.db.start()
                break
            except Exception as e:
                if i == retries - 1:
                    logging.error('Unable to initialize database container.')
                    raise e
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()