from google.cloud import dialogflowcx_v3beta1

def sample_list_versions():
    if False:
        for i in range(10):
            print('nop')
    client = dialogflowcx_v3beta1.VersionsClient()
    request = dialogflowcx_v3beta1.ListVersionsRequest(parent='parent_value')
    page_result = client.list_versions(request=request)
    for response in page_result:
        print(response)