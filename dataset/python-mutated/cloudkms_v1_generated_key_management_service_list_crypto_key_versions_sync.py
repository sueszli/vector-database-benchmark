from google.cloud import kms_v1

def sample_list_crypto_key_versions():
    if False:
        i = 10
        return i + 15
    client = kms_v1.KeyManagementServiceClient()
    request = kms_v1.ListCryptoKeyVersionsRequest(parent='parent_value')
    page_result = client.list_crypto_key_versions(request=request)
    for response in page_result:
        print(response)