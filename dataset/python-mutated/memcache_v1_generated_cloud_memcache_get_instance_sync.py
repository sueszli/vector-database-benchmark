from google.cloud import memcache_v1

def sample_get_instance():
    if False:
        i = 10
        return i + 15
    client = memcache_v1.CloudMemcacheClient()
    request = memcache_v1.GetInstanceRequest(name='name_value')
    response = client.get_instance(request=request)
    print(response)