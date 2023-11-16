from google.cloud import beyondcorp_clientconnectorservices_v1

def sample_get_client_connector_service():
    if False:
        i = 10
        return i + 15
    client = beyondcorp_clientconnectorservices_v1.ClientConnectorServicesServiceClient()
    request = beyondcorp_clientconnectorservices_v1.GetClientConnectorServiceRequest(name='name_value')
    response = client.get_client_connector_service(request=request)
    print(response)