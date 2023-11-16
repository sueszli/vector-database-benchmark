from google.cloud import commerce_consumer_procurement_v1alpha1

def sample_get_order():
    if False:
        return 10
    client = commerce_consumer_procurement_v1alpha1.ConsumerProcurementServiceClient()
    request = commerce_consumer_procurement_v1alpha1.GetOrderRequest(name='name_value')
    response = client.get_order(request=request)
    print(response)