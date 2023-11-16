from google.cloud import redis_v1beta1

def sample_failover_instance():
    if False:
        while True:
            i = 10
    client = redis_v1beta1.CloudRedisClient()
    request = redis_v1beta1.FailoverInstanceRequest(name='name_value')
    operation = client.failover_instance(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)