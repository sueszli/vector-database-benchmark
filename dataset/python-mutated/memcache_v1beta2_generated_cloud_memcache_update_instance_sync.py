from google.cloud import memcache_v1beta2

def sample_update_instance():
    if False:
        print('Hello World!')
    client = memcache_v1beta2.CloudMemcacheClient()
    resource = memcache_v1beta2.Instance()
    resource.name = 'name_value'
    resource.node_count = 1070
    resource.node_config.cpu_count = 976
    resource.node_config.memory_size_mb = 1505
    request = memcache_v1beta2.UpdateInstanceRequest(resource=resource)
    operation = client.update_instance(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)