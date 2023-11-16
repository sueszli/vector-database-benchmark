from typing import Optional
import psycopg2
from psycopg2 import sql
from feast.infra.registry.registry_store import RegistryStore
from feast.infra.utils.postgres.connection_utils import _get_conn
from feast.infra.utils.postgres.postgres_config import PostgreSQLConfig
from feast.protos.feast.core.Registry_pb2 import Registry as RegistryProto
from feast.repo_config import RegistryConfig

class PostgresRegistryConfig(RegistryConfig):
    host: str
    port: int
    database: str
    db_schema: str
    user: str
    password: str
    sslmode: Optional[str]
    sslkey_path: Optional[str]
    sslcert_path: Optional[str]
    sslrootcert_path: Optional[str]

class PostgreSQLRegistryStore(RegistryStore):

    def __init__(self, config: PostgresRegistryConfig, registry_path: str):
        if False:
            for i in range(10):
                print('nop')
        self.db_config = PostgreSQLConfig(host=config.host, port=config.port, database=config.database, db_schema=config.db_schema, user=config.user, password=config.password, sslmode=getattr(config, 'sslmode', None), sslkey_path=getattr(config, 'sslkey_path', None), sslcert_path=getattr(config, 'sslcert_path', None), sslrootcert_path=getattr(config, 'sslrootcert_path', None))
        self.table_name = config.path
        self.cache_ttl_seconds = config.cache_ttl_seconds

    def get_registry_proto(self) -> RegistryProto:
        if False:
            print('Hello World!')
        registry_proto = RegistryProto()
        try:
            with _get_conn(self.db_config) as conn, conn.cursor() as cur:
                cur.execute(sql.SQL('\n                        SELECT registry\n                        FROM {}\n                        WHERE version = (SELECT max(version) FROM {})\n                        ').format(sql.Identifier(self.table_name), sql.Identifier(self.table_name)))
                row = cur.fetchone()
                if row:
                    registry_proto = registry_proto.FromString(row[0])
        except psycopg2.errors.UndefinedTable:
            pass
        return registry_proto

    def update_registry_proto(self, registry_proto: RegistryProto):
        if False:
            print('Hello World!')
        '\n        Overwrites the current registry proto with the proto passed in. This method\n        writes to the registry path.\n\n        Args:\n            registry_proto: the new RegistryProto\n        '
        schema_name = self.db_config.db_schema or self.db_config.user
        with _get_conn(self.db_config) as conn, conn.cursor() as cur:
            cur.execute('\n                SELECT schema_name\n                FROM information_schema.schemata\n                WHERE schema_name = %s\n                ', (schema_name,))
            schema_exists = cur.fetchone()
            if not schema_exists:
                cur.execute(sql.SQL('CREATE SCHEMA IF NOT EXISTS {} AUTHORIZATION {}').format(sql.Identifier(schema_name), sql.Identifier(self.db_config.user)))
            cur.execute(sql.SQL('\n                    CREATE TABLE IF NOT EXISTS {} (\n                        version BIGSERIAL PRIMARY KEY,\n                        registry BYTEA NOT NULL\n                    );\n                    ').format(sql.Identifier(self.table_name)))
            cur.execute(sql.SQL('\n                    INSERT INTO {} (registry)\n                    VALUES (%s);\n                    ').format(sql.Identifier(self.table_name)), [registry_proto.SerializeToString()])

    def teardown(self):
        if False:
            for i in range(10):
                print('nop')
        with _get_conn(self.db_config) as conn, conn.cursor() as cur:
            cur.execute(sql.SQL('\n                    DROP TABLE IF EXISTS {};\n                    ').format(sql.Identifier(self.table_name)))