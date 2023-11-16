from google.cloud import dlp_v2

def sample_list_discovery_configs():
    if False:
        for i in range(10):
            print('nop')
    client = dlp_v2.DlpServiceClient()
    request = dlp_v2.ListDiscoveryConfigsRequest(parent='parent_value')
    page_result = client.list_discovery_configs(request=request)
    for response in page_result:
        print(response)