import google.api_core.exceptions
from google.cloud import bigquery_connection_v1 as bq_connection
from google.cloud.bigquery_connection_v1.services import connection_service
import pytest
import test_utils.prefixer
from . import create_mysql_connection
connection_prefixer = test_utils.prefixer.Prefixer('py-bq-r', 'snippets', separator='-')

@pytest.fixture(scope='session')
def location_path(connection_client: connection_service.ConnectionServiceClient(), project_id: str, location: str) -> str:
    if False:
        return 10
    return connection_client.common_location_path(project_id, location)

@pytest.fixture(scope='module', autouse=True)
def cleanup_connection(connection_client: connection_service.ConnectionServiceClient, location_path: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    for connection in connection_client.list_connections(parent=location_path):
        connection_id = connection.name.split('/')[-1]
        if connection_prefixer.should_cleanup(connection_id):
            connection_client.delete_connection(name=connection.name)

@pytest.fixture(scope='session')
def connection_id(connection_client: connection_service.ConnectionServiceClient, project_id: str, location: str) -> str:
    if False:
        print('Hello World!')
    id_ = connection_prefixer.create_prefix()
    yield id_
    connection_name = connection_client.connection_path(project_id, location, id_)
    try:
        connection_client.delete_connection(name=connection_name)
    except google.api_core.exceptions.NotFound:
        pass

@pytest.mark.parametrize('transport', ['grpc', 'rest'])
def test_create_mysql_connection(capsys: pytest.CaptureFixture, mysql_username: str, mysql_password: str, database: str, cloud_sql_conn_name: str, project_id: str, location: str, transport: str) -> None:
    if False:
        print('Hello World!')
    cloud_sql_credential = bq_connection.CloudSqlCredential({'username': mysql_username, 'password': mysql_password})
    cloud_sql_properties = bq_connection.CloudSqlProperties({'type_': bq_connection.CloudSqlProperties.DatabaseType.MYSQL, 'database': database, 'instance_id': cloud_sql_conn_name, 'credential': cloud_sql_credential})
    create_mysql_connection.create_mysql_connection(project_id=project_id, location=location, cloud_sql_properties=cloud_sql_properties, transport=transport)
    (out, _) = capsys.readouterr()
    assert 'Created connection successfully:' in out