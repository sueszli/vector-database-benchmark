from google.cloud import dialogflow_v2

def sample_list_versions():
    if False:
        i = 10
        return i + 15
    client = dialogflow_v2.VersionsClient()
    request = dialogflow_v2.ListVersionsRequest(parent='parent_value')
    page_result = client.list_versions(request=request)
    for response in page_result:
        print(response)