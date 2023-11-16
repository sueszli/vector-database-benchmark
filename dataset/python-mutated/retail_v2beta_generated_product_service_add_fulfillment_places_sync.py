from google.cloud import retail_v2beta

def sample_add_fulfillment_places():
    if False:
        return 10
    client = retail_v2beta.ProductServiceClient()
    request = retail_v2beta.AddFulfillmentPlacesRequest(product='product_value', type_='type__value', place_ids=['place_ids_value1', 'place_ids_value2'])
    operation = client.add_fulfillment_places(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)