from google.cloud import dataplex_v1

def sample_list_asset_actions():
    if False:
        for i in range(10):
            print('nop')
    client = dataplex_v1.DataplexServiceClient()
    request = dataplex_v1.ListAssetActionsRequest(parent='parent_value')
    page_result = client.list_asset_actions(request=request)
    for response in page_result:
        print(response)