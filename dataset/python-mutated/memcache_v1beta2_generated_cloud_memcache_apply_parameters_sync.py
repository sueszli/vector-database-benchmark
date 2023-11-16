from google.cloud import memcache_v1beta2

def sample_apply_parameters():
    if False:
        return 10
    client = memcache_v1beta2.CloudMemcacheClient()
    request = memcache_v1beta2.ApplyParametersRequest(name='name_value')
    operation = client.apply_parameters(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)