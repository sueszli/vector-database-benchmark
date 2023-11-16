from google.shopping import merchant_inventories_v1beta

def sample_list_regional_inventories():
    if False:
        while True:
            i = 10
    client = merchant_inventories_v1beta.RegionalInventoryServiceClient()
    request = merchant_inventories_v1beta.ListRegionalInventoriesRequest(parent='parent_value')
    page_result = client.list_regional_inventories(request=request)
    for response in page_result:
        print(response)