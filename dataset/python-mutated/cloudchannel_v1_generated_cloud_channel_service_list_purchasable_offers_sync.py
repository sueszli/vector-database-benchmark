from google.cloud import channel_v1

def sample_list_purchasable_offers():
    if False:
        for i in range(10):
            print('nop')
    client = channel_v1.CloudChannelServiceClient()
    create_entitlement_purchase = channel_v1.CreateEntitlementPurchase()
    create_entitlement_purchase.sku = 'sku_value'
    request = channel_v1.ListPurchasableOffersRequest(create_entitlement_purchase=create_entitlement_purchase, customer='customer_value')
    page_result = client.list_purchasable_offers(request=request)
    for response in page_result:
        print(response)