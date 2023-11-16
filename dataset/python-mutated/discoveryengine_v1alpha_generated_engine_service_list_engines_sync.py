from google.cloud import discoveryengine_v1alpha

def sample_list_engines():
    if False:
        i = 10
        return i + 15
    client = discoveryengine_v1alpha.EngineServiceClient()
    request = discoveryengine_v1alpha.ListEnginesRequest(parent='parent_value')
    page_result = client.list_engines(request=request)
    for response in page_result:
        print(response)