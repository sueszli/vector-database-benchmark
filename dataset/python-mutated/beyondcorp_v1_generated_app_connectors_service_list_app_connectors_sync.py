from google.cloud import beyondcorp_appconnectors_v1

def sample_list_app_connectors():
    if False:
        for i in range(10):
            print('nop')
    client = beyondcorp_appconnectors_v1.AppConnectorsServiceClient()
    request = beyondcorp_appconnectors_v1.ListAppConnectorsRequest(parent='parent_value')
    page_result = client.list_app_connectors(request=request)
    for response in page_result:
        print(response)