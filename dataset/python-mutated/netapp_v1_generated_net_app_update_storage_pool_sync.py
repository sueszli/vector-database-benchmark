from google.cloud import netapp_v1

def sample_update_storage_pool():
    if False:
        return 10
    client = netapp_v1.NetAppClient()
    storage_pool = netapp_v1.StoragePool()
    storage_pool.service_level = 'STANDARD'
    storage_pool.capacity_gib = 1247
    storage_pool.network = 'network_value'
    request = netapp_v1.UpdateStoragePoolRequest(storage_pool=storage_pool)
    operation = client.update_storage_pool(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)