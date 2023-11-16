from google.cloud import dialogflowcx_v3

def sample_list_versions():
    if False:
        return 10
    client = dialogflowcx_v3.VersionsClient()
    request = dialogflowcx_v3.ListVersionsRequest(parent='parent_value')
    page_result = client.list_versions(request=request)
    for response in page_result:
        print(response)