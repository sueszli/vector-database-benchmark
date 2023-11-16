from google.cloud import retail_v2alpha

def sample_complete_query():
    if False:
        print('Hello World!')
    client = retail_v2alpha.CompletionServiceClient()
    request = retail_v2alpha.CompleteQueryRequest(catalog='catalog_value', query='query_value')
    response = client.complete_query(request=request)
    print(response)