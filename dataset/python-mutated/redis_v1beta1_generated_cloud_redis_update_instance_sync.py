from google.cloud import redis_v1beta1

def sample_update_instance():
    if False:
        print('Hello World!')
    client = redis_v1beta1.CloudRedisClient()
    instance = redis_v1beta1.Instance()
    instance.name = 'name_value'
    instance.tier = 'STANDARD_HA'
    instance.memory_size_gb = 1499
    request = redis_v1beta1.UpdateInstanceRequest(instance=instance)
    operation = client.update_instance(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)