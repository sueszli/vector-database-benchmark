from google.cloud import discoveryengine_v1alpha

def sample_complete_query():
    if False:
        while True:
            i = 10
    client = discoveryengine_v1alpha.CompletionServiceClient()
    request = discoveryengine_v1alpha.CompleteQueryRequest(data_store='data_store_value', query='query_value')
    response = client.complete_query(request=request)
    print(response)