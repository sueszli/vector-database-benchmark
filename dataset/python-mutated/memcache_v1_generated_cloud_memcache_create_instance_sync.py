from google.cloud import memcache_v1

def sample_create_instance():
    if False:
        return 10
    client = memcache_v1.CloudMemcacheClient()
    instance = memcache_v1.Instance()
    instance.name = 'name_value'
    instance.node_count = 1070
    instance.node_config.cpu_count = 976
    instance.node_config.memory_size_mb = 1505
    request = memcache_v1.CreateInstanceRequest(parent='parent_value', instance_id='instance_id_value', instance=instance)
    operation = client.create_instance(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)