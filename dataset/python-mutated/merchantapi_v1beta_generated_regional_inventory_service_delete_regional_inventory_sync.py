from google.shopping import merchant_inventories_v1beta

def sample_delete_regional_inventory():
    if False:
        return 10
    client = merchant_inventories_v1beta.RegionalInventoryServiceClient()
    request = merchant_inventories_v1beta.DeleteRegionalInventoryRequest(name='name_value')
    client.delete_regional_inventory(request=request)