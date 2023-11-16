from google.cloud import commerce_consumer_procurement_v1

def sample_list_orders():
    if False:
        while True:
            i = 10
    client = commerce_consumer_procurement_v1.ConsumerProcurementServiceClient()
    request = commerce_consumer_procurement_v1.ListOrdersRequest(parent='parent_value')
    page_result = client.list_orders(request=request)
    for response in page_result:
        print(response)