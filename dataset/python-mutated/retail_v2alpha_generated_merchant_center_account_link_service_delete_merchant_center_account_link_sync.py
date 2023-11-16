from google.cloud import retail_v2alpha

def sample_delete_merchant_center_account_link():
    if False:
        for i in range(10):
            print('nop')
    client = retail_v2alpha.MerchantCenterAccountLinkServiceClient()
    request = retail_v2alpha.DeleteMerchantCenterAccountLinkRequest(name='name_value')
    client.delete_merchant_center_account_link(request=request)