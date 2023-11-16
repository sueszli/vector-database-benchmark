from google.cloud import memcache_v1beta2

def sample_update_parameters():
    if False:
        print('Hello World!')
    client = memcache_v1beta2.CloudMemcacheClient()
    request = memcache_v1beta2.UpdateParametersRequest(name='name_value')
    operation = client.update_parameters(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)