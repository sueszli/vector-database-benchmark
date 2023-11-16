from google.cloud import retail_v2alpha

def sample_search():
    if False:
        return 10
    client = retail_v2alpha.SearchServiceClient()
    request = retail_v2alpha.SearchRequest(placement='placement_value', visitor_id='visitor_id_value')
    page_result = client.search(request=request)
    for response in page_result:
        print(response)