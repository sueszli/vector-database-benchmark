from google.cloud import retail_v2

def sample_set_inventory():
    if False:
        return 10
    client = retail_v2.ProductServiceClient()
    inventory = retail_v2.Product()
    inventory.title = 'title_value'
    request = retail_v2.SetInventoryRequest(inventory=inventory)
    operation = client.set_inventory(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)