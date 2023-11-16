from google.cloud import retail_v2

def sample_add_fulfillment_places():
    if False:
        return 10
    client = retail_v2.ProductServiceClient()
    request = retail_v2.AddFulfillmentPlacesRequest(product='product_value', type_='type__value', place_ids=['place_ids_value1', 'place_ids_value2'])
    operation = client.add_fulfillment_places(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)