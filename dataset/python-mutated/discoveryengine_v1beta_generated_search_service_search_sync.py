from google.cloud import discoveryengine_v1beta

def sample_search():
    if False:
        for i in range(10):
            print('nop')
    client = discoveryengine_v1beta.SearchServiceClient()
    request = discoveryengine_v1beta.SearchRequest(serving_config='serving_config_value')
    page_result = client.search(request=request)
    for response in page_result:
        print(response)