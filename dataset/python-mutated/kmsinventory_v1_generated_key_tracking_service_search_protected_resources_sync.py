from google.cloud import kms_inventory_v1

def sample_search_protected_resources():
    if False:
        for i in range(10):
            print('nop')
    client = kms_inventory_v1.KeyTrackingServiceClient()
    request = kms_inventory_v1.SearchProtectedResourcesRequest(scope='scope_value', crypto_key='crypto_key_value')
    page_result = client.search_protected_resources(request=request)
    for response in page_result:
        print(response)