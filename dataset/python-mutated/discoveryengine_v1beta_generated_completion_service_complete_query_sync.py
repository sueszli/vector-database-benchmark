from google.cloud import discoveryengine_v1beta

def sample_complete_query():
    if False:
        print('Hello World!')
    client = discoveryengine_v1beta.CompletionServiceClient()
    request = discoveryengine_v1beta.CompleteQueryRequest(data_store='data_store_value', query='query_value')
    response = client.complete_query(request=request)
    print(response)