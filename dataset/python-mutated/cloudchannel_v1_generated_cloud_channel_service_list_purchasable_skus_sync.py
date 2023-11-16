from google.cloud import channel_v1

def sample_list_purchasable_skus():
    if False:
        print('Hello World!')
    client = channel_v1.CloudChannelServiceClient()
    create_entitlement_purchase = channel_v1.CreateEntitlementPurchase()
    create_entitlement_purchase.product = 'product_value'
    request = channel_v1.ListPurchasableSkusRequest(create_entitlement_purchase=create_entitlement_purchase, customer='customer_value')
    page_result = client.list_purchasable_skus(request=request)
    for response in page_result:
        print(response)