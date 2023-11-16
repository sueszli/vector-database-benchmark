from google.cloud import bigquery_connection_v1 as bq_connection
'This sample shows how to create a BigQuery connection with a Cloud SQL for MySQL database'

def main() -> None:
    if False:
        i = 10
        return i + 15
    project_id = 'your-project-id'
    location = 'US'
    database = 'my-database'
    username = 'my-username'
    password = 'my-password'
    cloud_sql_conn_name = ''
    transport = 'grpc'
    cloud_sql_credential = bq_connection.CloudSqlCredential({'username': username, 'password': password})
    cloud_sql_properties = bq_connection.CloudSqlProperties({'type_': bq_connection.CloudSqlProperties.DatabaseType.MYSQL, 'database': database, 'instance_id': cloud_sql_conn_name, 'credential': cloud_sql_credential})
    create_mysql_connection(project_id, location, cloud_sql_properties, transport)

def create_mysql_connection(project_id: str, location: str, cloud_sql_properties: bq_connection.CloudSqlProperties, transport: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    connection = bq_connection.types.Connection({'cloud_sql': cloud_sql_properties})
    client = bq_connection.ConnectionServiceClient(transport=transport)
    parent = client.common_location_path(project_id, location)
    request = bq_connection.CreateConnectionRequest({'parent': parent, 'connection': connection})
    response = client.create_connection(request)
    print(f'Created connection successfully: {response.name}')