from google.cloud import netapp_v1

def sample_get_storage_pool():
    if False:
        while True:
            i = 10
    client = netapp_v1.NetAppClient()
    request = netapp_v1.GetStoragePoolRequest(name='name_value')
    response = client.get_storage_pool(request=request)
    print(response)