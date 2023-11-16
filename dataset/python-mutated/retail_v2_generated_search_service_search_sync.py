from google.cloud import retail_v2

def sample_search():
    if False:
        print('Hello World!')
    client = retail_v2.SearchServiceClient()
    request = retail_v2.SearchRequest(placement='placement_value', visitor_id='visitor_id_value')
    page_result = client.search(request=request)
    for response in page_result:
        print(response)