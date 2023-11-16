from google.cloud import beyondcorp_clientconnectorservices_v1

def sample_list_client_connector_services():
    if False:
        return 10
    client = beyondcorp_clientconnectorservices_v1.ClientConnectorServicesServiceClient()
    request = beyondcorp_clientconnectorservices_v1.ListClientConnectorServicesRequest(parent='parent_value')
    page_result = client.list_client_connector_services(request=request)
    for response in page_result:
        print(response)