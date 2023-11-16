from google.cloud import memcache_v1beta2

def sample_apply_software_update():
    if False:
        for i in range(10):
            print('nop')
    client = memcache_v1beta2.CloudMemcacheClient()
    request = memcache_v1beta2.ApplySoftwareUpdateRequest(instance='instance_value')
    operation = client.apply_software_update(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)