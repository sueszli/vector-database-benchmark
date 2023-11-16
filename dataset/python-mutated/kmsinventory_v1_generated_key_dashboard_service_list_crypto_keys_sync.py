from google.cloud import kms_inventory_v1

def sample_list_crypto_keys():
    if False:
        for i in range(10):
            print('nop')
    client = kms_inventory_v1.KeyDashboardServiceClient()
    request = kms_inventory_v1.ListCryptoKeysRequest(parent='parent_value')
    page_result = client.list_crypto_keys(request=request)
    for response in page_result:
        print(response)