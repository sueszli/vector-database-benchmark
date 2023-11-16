from google.cloud import memcache_v1beta2

def sample_get_instance():
    if False:
        return 10
    client = memcache_v1beta2.CloudMemcacheClient()
    request = memcache_v1beta2.GetInstanceRequest(name='name_value')
    response = client.get_instance(request=request)
    print(response)