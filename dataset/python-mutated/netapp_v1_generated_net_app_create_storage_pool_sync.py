from google.cloud import netapp_v1

def sample_create_storage_pool():
    if False:
        i = 10
        return i + 15
    client = netapp_v1.NetAppClient()
    storage_pool = netapp_v1.StoragePool()
    storage_pool.service_level = 'STANDARD'
    storage_pool.capacity_gib = 1247
    storage_pool.network = 'network_value'
    request = netapp_v1.CreateStoragePoolRequest(parent='parent_value', storage_pool_id='storage_pool_id_value', storage_pool=storage_pool)
    operation = client.create_storage_pool(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)