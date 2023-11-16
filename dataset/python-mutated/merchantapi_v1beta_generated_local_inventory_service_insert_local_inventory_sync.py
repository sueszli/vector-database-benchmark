from google.shopping import merchant_inventories_v1beta

def sample_insert_local_inventory():
    if False:
        print('Hello World!')
    client = merchant_inventories_v1beta.LocalInventoryServiceClient()
    local_inventory = merchant_inventories_v1beta.LocalInventory()
    local_inventory.store_code = 'store_code_value'
    request = merchant_inventories_v1beta.InsertLocalInventoryRequest(parent='parent_value', local_inventory=local_inventory)
    response = client.insert_local_inventory(request=request)
    print(response)