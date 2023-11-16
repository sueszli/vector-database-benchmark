from google.cloud import retail_v2beta

def sample_add_local_inventories():
    if False:
        print('Hello World!')
    client = retail_v2beta.ProductServiceClient()
    request = retail_v2beta.AddLocalInventoriesRequest(product='product_value')
    operation = client.add_local_inventories(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)