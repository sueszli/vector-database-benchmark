from google.cloud import discoveryengine_v1

def sample_complete_query():
    if False:
        while True:
            i = 10
    client = discoveryengine_v1.CompletionServiceClient()
    request = discoveryengine_v1.CompleteQueryRequest(data_store='data_store_value', query='query_value')
    response = client.complete_query(request=request)
    print(response)