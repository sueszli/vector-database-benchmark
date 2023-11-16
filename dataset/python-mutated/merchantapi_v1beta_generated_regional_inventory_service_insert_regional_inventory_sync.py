from google.shopping import merchant_inventories_v1beta

def sample_insert_regional_inventory():
    if False:
        return 10
    client = merchant_inventories_v1beta.RegionalInventoryServiceClient()
    regional_inventory = merchant_inventories_v1beta.RegionalInventory()
    regional_inventory.region = 'region_value'
    request = merchant_inventories_v1beta.InsertRegionalInventoryRequest(parent='parent_value', regional_inventory=regional_inventory)
    response = client.insert_regional_inventory(request=request)
    print(response)