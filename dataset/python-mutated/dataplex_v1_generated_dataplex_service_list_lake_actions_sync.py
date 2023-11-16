from google.cloud import dataplex_v1

def sample_list_lake_actions():
    if False:
        return 10
    client = dataplex_v1.DataplexServiceClient()
    request = dataplex_v1.ListLakeActionsRequest(parent='parent_value')
    page_result = client.list_lake_actions(request=request)
    for response in page_result:
        print(response)