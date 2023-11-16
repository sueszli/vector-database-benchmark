from google.cloud import retail_v2alpha

def sample_set_inventory():
    if False:
        while True:
            i = 10
    client = retail_v2alpha.ProductServiceClient()
    inventory = retail_v2alpha.Product()
    inventory.title = 'title_value'
    request = retail_v2alpha.SetInventoryRequest(inventory=inventory)
    operation = client.set_inventory(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)