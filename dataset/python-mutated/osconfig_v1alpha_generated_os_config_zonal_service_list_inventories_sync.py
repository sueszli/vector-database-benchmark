from google.cloud import osconfig_v1alpha

def sample_list_inventories():
    if False:
        print('Hello World!')
    client = osconfig_v1alpha.OsConfigZonalServiceClient()
    request = osconfig_v1alpha.ListInventoriesRequest(parent='parent_value')
    page_result = client.list_inventories(request=request)
    for response in page_result:
        print(response)