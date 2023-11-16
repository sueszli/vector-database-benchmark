from google.cloud import redis_v1

def sample_delete_instance():
    if False:
        while True:
            i = 10
    client = redis_v1.CloudRedisClient()
    request = redis_v1.DeleteInstanceRequest(name='name_value')
    operation = client.delete_instance(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)