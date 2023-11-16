from google.cloud import retail_v2beta

def sample_search():
    if False:
        for i in range(10):
            print('nop')
    client = retail_v2beta.SearchServiceClient()
    request = retail_v2beta.SearchRequest(placement='placement_value', visitor_id='visitor_id_value')
    page_result = client.search(request=request)
    for response in page_result:
        print(response)