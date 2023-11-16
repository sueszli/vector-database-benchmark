from google.cloud import memcache_v1

def sample_update_instance():
    if False:
        for i in range(10):
            print('nop')
    client = memcache_v1.CloudMemcacheClient()
    instance = memcache_v1.Instance()
    instance.name = 'name_value'
    instance.node_count = 1070
    instance.node_config.cpu_count = 976
    instance.node_config.memory_size_mb = 1505
    request = memcache_v1.UpdateInstanceRequest(instance=instance)
    operation = client.update_instance(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)