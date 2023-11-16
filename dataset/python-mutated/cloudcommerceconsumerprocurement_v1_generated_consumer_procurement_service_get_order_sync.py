from google.cloud import commerce_consumer_procurement_v1

def sample_get_order():
    if False:
        for i in range(10):
            print('nop')
    client = commerce_consumer_procurement_v1.ConsumerProcurementServiceClient()
    request = commerce_consumer_procurement_v1.GetOrderRequest(name='name_value')
    response = client.get_order(request=request)
    print(response)