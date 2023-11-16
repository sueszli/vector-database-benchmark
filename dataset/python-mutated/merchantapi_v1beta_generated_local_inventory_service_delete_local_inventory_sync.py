from google.shopping import merchant_inventories_v1beta

def sample_delete_local_inventory():
    if False:
        i = 10
        return i + 15
    client = merchant_inventories_v1beta.LocalInventoryServiceClient()
    request = merchant_inventories_v1beta.DeleteLocalInventoryRequest(name='name_value')
    client.delete_local_inventory(request=request)