from google.cloud import talent_v4

def sample_complete_query():
    if False:
        print('Hello World!')
    client = talent_v4.CompletionClient()
    request = talent_v4.CompleteQueryRequest(tenant='tenant_value', query='query_value', page_size=951)
    response = client.complete_query(request=request)
    print(response)