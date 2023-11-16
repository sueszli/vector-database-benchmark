from google.cloud import commerce_consumer_procurement_v1alpha1

def sample_place_order():
    if False:
        return 10
    client = commerce_consumer_procurement_v1alpha1.ConsumerProcurementServiceClient()
    request = commerce_consumer_procurement_v1alpha1.PlaceOrderRequest(parent='parent_value', display_name='display_name_value')
    operation = client.place_order(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)