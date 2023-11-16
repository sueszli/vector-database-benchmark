from google.cloud import retail_v2alpha

def sample_list_merchant_center_account_links():
    if False:
        for i in range(10):
            print('nop')
    client = retail_v2alpha.MerchantCenterAccountLinkServiceClient()
    request = retail_v2alpha.ListMerchantCenterAccountLinksRequest(parent='parent_value')
    response = client.list_merchant_center_account_links(request=request)
    print(response)