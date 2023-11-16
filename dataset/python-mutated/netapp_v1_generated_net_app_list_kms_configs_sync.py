from google.cloud import netapp_v1

def sample_list_kms_configs():
    if False:
        while True:
            i = 10
    client = netapp_v1.NetAppClient()
    request = netapp_v1.ListKmsConfigsRequest(parent='parent_value')
    page_result = client.list_kms_configs(request=request)
    for response in page_result:
        print(response)