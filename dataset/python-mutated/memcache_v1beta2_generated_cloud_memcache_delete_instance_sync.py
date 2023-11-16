from google.cloud import memcache_v1beta2

def sample_delete_instance():
    if False:
        i = 10
        return i + 15
    client = memcache_v1beta2.CloudMemcacheClient()
    request = memcache_v1beta2.DeleteInstanceRequest(name='name_value')
    operation = client.delete_instance(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)