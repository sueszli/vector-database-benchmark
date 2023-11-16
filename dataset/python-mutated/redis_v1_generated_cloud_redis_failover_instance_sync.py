from google.cloud import redis_v1

def sample_failover_instance():
    if False:
        return 10
    client = redis_v1.CloudRedisClient()
    request = redis_v1.FailoverInstanceRequest(name='name_value')
    operation = client.failover_instance(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)