from google.cloud import dataplex_v1

def sample_list_zones():
    if False:
        print('Hello World!')
    client = dataplex_v1.DataplexServiceClient()
    request = dataplex_v1.ListZonesRequest(parent='parent_value')
    page_result = client.list_zones(request=request)
    for response in page_result:
        print(response)