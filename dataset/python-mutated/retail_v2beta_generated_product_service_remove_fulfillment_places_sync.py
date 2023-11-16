from google.cloud import retail_v2beta

def sample_remove_fulfillment_places():
    if False:
        for i in range(10):
            print('nop')
    client = retail_v2beta.ProductServiceClient()
    request = retail_v2beta.RemoveFulfillmentPlacesRequest(product='product_value', type_='type__value', place_ids=['place_ids_value1', 'place_ids_value2'])
    operation = client.remove_fulfillment_places(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)