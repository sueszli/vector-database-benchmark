from google.cloud import retail_v2alpha

def sample_remove_local_inventories():
    if False:
        for i in range(10):
            print('nop')
    client = retail_v2alpha.ProductServiceClient()
    request = retail_v2alpha.RemoveLocalInventoriesRequest(product='product_value', place_ids=['place_ids_value1', 'place_ids_value2'])
    operation = client.remove_local_inventories(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)