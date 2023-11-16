from google.cloud import memcache_v1

def sample_apply_parameters():
    if False:
        while True:
            i = 10
    client = memcache_v1.CloudMemcacheClient()
    request = memcache_v1.ApplyParametersRequest(name='name_value')
    operation = client.apply_parameters(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)