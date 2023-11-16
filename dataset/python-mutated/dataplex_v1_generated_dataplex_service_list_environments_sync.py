from google.cloud import dataplex_v1

def sample_list_environments():
    if False:
        i = 10
        return i + 15
    client = dataplex_v1.DataplexServiceClient()
    request = dataplex_v1.ListEnvironmentsRequest(parent='parent_value')
    page_result = client.list_environments(request=request)
    for response in page_result:
        print(response)