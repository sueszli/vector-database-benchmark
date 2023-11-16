from google.cloud import beyondcorp_appconnections_v1

def sample_create_app_connection():
    if False:
        print('Hello World!')
    client = beyondcorp_appconnections_v1.AppConnectionsServiceClient()
    app_connection = beyondcorp_appconnections_v1.AppConnection()
    app_connection.name = 'name_value'
    app_connection.type_ = 'TCP_PROXY'
    app_connection.application_endpoint.host = 'host_value'
    app_connection.application_endpoint.port = 453
    request = beyondcorp_appconnections_v1.CreateAppConnectionRequest(parent='parent_value', app_connection=app_connection)
    operation = client.create_app_connection(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)