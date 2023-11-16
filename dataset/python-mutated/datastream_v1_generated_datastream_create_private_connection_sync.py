from google.cloud import datastream_v1

def sample_create_private_connection():
    if False:
        print('Hello World!')
    client = datastream_v1.DatastreamClient()
    private_connection = datastream_v1.PrivateConnection()
    private_connection.display_name = 'display_name_value'
    request = datastream_v1.CreatePrivateConnectionRequest(parent='parent_value', private_connection_id='private_connection_id_value', private_connection=private_connection)
    operation = client.create_private_connection(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)