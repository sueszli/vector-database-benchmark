from google.cloud import redis_v1

def sample_get_instance():
    if False:
        while True:
            i = 10
    client = redis_v1.CloudRedisClient()
    request = redis_v1.GetInstanceRequest(name='name_value')
    response = client.get_instance(request=request)
    print(response)