from google.cloud import dataplex_v1

def sample_list_zone_actions():
    if False:
        return 10
    client = dataplex_v1.DataplexServiceClient()
    request = dataplex_v1.ListZoneActionsRequest(parent='parent_value')
    page_result = client.list_zone_actions(request=request)
    for response in page_result:
        print(response)