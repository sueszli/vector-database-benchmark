from google.cloud import beyondcorp_clientconnectorservices_v1

def sample_delete_client_connector_service():
    if False:
        print('Hello World!')
    client = beyondcorp_clientconnectorservices_v1.ClientConnectorServicesServiceClient()
    request = beyondcorp_clientconnectorservices_v1.DeleteClientConnectorServiceRequest(name='name_value')
    operation = client.delete_client_connector_service(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)