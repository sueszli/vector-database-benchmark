from google.cloud import data_fusion_v1

def sample_list_available_versions():
    if False:
        while True:
            i = 10
    client = data_fusion_v1.DataFusionClient()
    request = data_fusion_v1.ListAvailableVersionsRequest(parent='parent_value')
    page_result = client.list_available_versions(request=request)
    for response in page_result:
        print(response)