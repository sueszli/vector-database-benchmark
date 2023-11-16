from google.cloud import retail_v2beta

def sample_complete_query():
    if False:
        return 10
    client = retail_v2beta.CompletionServiceClient()
    request = retail_v2beta.CompleteQueryRequest(catalog='catalog_value', query='query_value')
    response = client.complete_query(request=request)
    print(response)