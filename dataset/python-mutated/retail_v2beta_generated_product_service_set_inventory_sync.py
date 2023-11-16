from google.cloud import retail_v2beta

def sample_set_inventory():
    if False:
        return 10
    client = retail_v2beta.ProductServiceClient()
    inventory = retail_v2beta.Product()
    inventory.title = 'title_value'
    request = retail_v2beta.SetInventoryRequest(inventory=inventory)
    operation = client.set_inventory(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)