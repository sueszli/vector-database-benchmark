from google.cloud import memcache_v1

def sample_update_parameters():
    if False:
        return 10
    client = memcache_v1.CloudMemcacheClient()
    request = memcache_v1.UpdateParametersRequest(name='name_value')
    operation = client.update_parameters(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)