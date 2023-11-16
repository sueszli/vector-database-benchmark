from google.shopping import merchant_inventories_v1beta

def sample_list_local_inventories():
    if False:
        for i in range(10):
            print('nop')
    client = merchant_inventories_v1beta.LocalInventoryServiceClient()
    request = merchant_inventories_v1beta.ListLocalInventoriesRequest(parent='parent_value')
    page_result = client.list_local_inventories(request=request)
    for response in page_result:
        print(response)