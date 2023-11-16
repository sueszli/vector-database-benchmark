from google.cloud import osconfig_v1

def sample_list_inventories():
    if False:
        while True:
            i = 10
    client = osconfig_v1.OsConfigZonalServiceClient()
    request = osconfig_v1.ListInventoriesRequest(parent='parent_value')
    page_result = client.list_inventories(request=request)
    for response in page_result:
        print(response)