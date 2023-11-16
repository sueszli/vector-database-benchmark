from google.cloud import retail_v2alpha

def sample_create_merchant_center_account_link():
    if False:
        i = 10
        return i + 15
    client = retail_v2alpha.MerchantCenterAccountLinkServiceClient()
    merchant_center_account_link = retail_v2alpha.MerchantCenterAccountLink()
    merchant_center_account_link.merchant_center_account_id = 2730
    merchant_center_account_link.branch_id = 'branch_id_value'
    request = retail_v2alpha.CreateMerchantCenterAccountLinkRequest(parent='parent_value', merchant_center_account_link=merchant_center_account_link)
    operation = client.create_merchant_center_account_link(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)