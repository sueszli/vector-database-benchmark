from google.cloud import retail_v2

def sample_add_local_inventories():
    if False:
        i = 10
        return i + 15
    client = retail_v2.ProductServiceClient()
    request = retail_v2.AddLocalInventoriesRequest(product='product_value')
    operation = client.add_local_inventories(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)