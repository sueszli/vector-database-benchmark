from google.cloud import dataplex_v1

def sample_list_sessions():
    if False:
        return 10
    client = dataplex_v1.DataplexServiceClient()
    request = dataplex_v1.ListSessionsRequest(parent='parent_value')
    page_result = client.list_sessions(request=request)
    for response in page_result:
        print(response)