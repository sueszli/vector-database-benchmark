from google.cloud import redis_v1beta1

def sample_create_instance():
    if False:
        while True:
            i = 10
    client = redis_v1beta1.CloudRedisClient()
    instance = redis_v1beta1.Instance()
    instance.name = 'name_value'
    instance.tier = 'STANDARD_HA'
    instance.memory_size_gb = 1499
    request = redis_v1beta1.CreateInstanceRequest(parent='parent_value', instance_id='instance_id_value', instance=instance)
    operation = client.create_instance(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)