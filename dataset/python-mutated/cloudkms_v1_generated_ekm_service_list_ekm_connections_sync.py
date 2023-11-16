from google.cloud import kms_v1

def sample_list_ekm_connections():
    if False:
        for i in range(10):
            print('nop')
    client = kms_v1.EkmServiceClient()
    request = kms_v1.ListEkmConnectionsRequest(parent='parent_value')
    page_result = client.list_ekm_connections(request=request)
    for response in page_result:
        print(response)