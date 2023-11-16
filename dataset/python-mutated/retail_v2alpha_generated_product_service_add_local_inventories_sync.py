from google.cloud import retail_v2alpha

def sample_add_local_inventories():
    if False:
        return 10
    client = retail_v2alpha.ProductServiceClient()
    request = retail_v2alpha.AddLocalInventoriesRequest(product='product_value')
    operation = client.add_local_inventories(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)