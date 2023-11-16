from google.cloud import talent_v4beta1

def sample_complete_query():
    if False:
        print('Hello World!')
    client = talent_v4beta1.CompletionClient()
    request = talent_v4beta1.CompleteQueryRequest(parent='parent_value', query='query_value', page_size=951)
    response = client.complete_query(request=request)
    print(response)