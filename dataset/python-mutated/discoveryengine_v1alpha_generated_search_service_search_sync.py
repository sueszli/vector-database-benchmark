from google.cloud import discoveryengine_v1alpha

def sample_search():
    if False:
        print('Hello World!')
    client = discoveryengine_v1alpha.SearchServiceClient()
    request = discoveryengine_v1alpha.SearchRequest(serving_config='serving_config_value')
    page_result = client.search(request=request)
    for response in page_result:
        print(response)