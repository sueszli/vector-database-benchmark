from google.cloud import kms_v1

def sample_list_key_rings():
    if False:
        return 10
    client = kms_v1.KeyManagementServiceClient()
    request = kms_v1.ListKeyRingsRequest(parent='parent_value')
    page_result = client.list_key_rings(request=request)
    for response in page_result:
        print(response)