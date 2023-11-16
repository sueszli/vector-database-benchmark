from google.cloud import dialogflow_v2beta1

def sample_list_versions():
    if False:
        print('Hello World!')
    client = dialogflow_v2beta1.VersionsClient()
    request = dialogflow_v2beta1.ListVersionsRequest(parent='parent_value')
    page_result = client.list_versions(request=request)
    for response in page_result:
        print(response)