from google.cloud import retail_v2alpha

def sample_remove_fulfillment_places():
    if False:
        print('Hello World!')
    client = retail_v2alpha.ProductServiceClient()
    request = retail_v2alpha.RemoveFulfillmentPlacesRequest(product='product_value', type_='type__value', place_ids=['place_ids_value1', 'place_ids_value2'])
    operation = client.remove_fulfillment_places(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)