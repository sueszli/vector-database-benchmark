from google.cloud import netapp_v1

def sample_list_storage_pools():
    if False:
        i = 10
        return i + 15
    client = netapp_v1.NetAppClient()
    request = netapp_v1.ListStoragePoolsRequest(parent='parent_value')
    page_result = client.list_storage_pools(request=request)
    for response in page_result:
        print(response)