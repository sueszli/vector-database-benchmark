from google.cloud import redis_v1

def sample_update_instance():
    if False:
        for i in range(10):
            print('nop')
    client = redis_v1.CloudRedisClient()
    instance = redis_v1.Instance()
    instance.name = 'name_value'
    instance.tier = 'STANDARD_HA'
    instance.memory_size_gb = 1499
    request = redis_v1.UpdateInstanceRequest(instance=instance)
    operation = client.update_instance(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)