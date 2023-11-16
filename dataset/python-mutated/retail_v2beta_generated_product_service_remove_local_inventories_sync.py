from google.cloud import retail_v2beta

def sample_remove_local_inventories():
    if False:
        i = 10
        return i + 15
    client = retail_v2beta.ProductServiceClient()
    request = retail_v2beta.RemoveLocalInventoriesRequest(product='product_value', place_ids=['place_ids_value1', 'place_ids_value2'])
    operation = client.remove_local_inventories(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)