from google.cloud import discoveryengine_v1

def sample_search():
    if False:
        print('Hello World!')
    client = discoveryengine_v1.SearchServiceClient()
    request = discoveryengine_v1.SearchRequest(serving_config='serving_config_value')
    page_result = client.search(request=request)
    for response in page_result:
        print(response)