from google.cloud import redis_v1

def sample_get_instance_auth_string():
    if False:
        for i in range(10):
            print('nop')
    client = redis_v1.CloudRedisClient()
    request = redis_v1.GetInstanceAuthStringRequest(name='name_value')
    response = client.get_instance_auth_string(request=request)
    print(response)