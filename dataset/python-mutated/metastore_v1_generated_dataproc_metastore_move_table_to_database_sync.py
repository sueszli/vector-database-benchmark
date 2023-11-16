from google.cloud import metastore_v1

def sample_move_table_to_database():
    if False:
        i = 10
        return i + 15
    client = metastore_v1.DataprocMetastoreClient()
    request = metastore_v1.MoveTableToDatabaseRequest(service='service_value', table_name='table_name_value', db_name='db_name_value', destination_db_name='destination_db_name_value')
    operation = client.move_table_to_database(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)