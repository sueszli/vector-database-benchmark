from google.cloud import redis_v1

def sample_upgrade_instance():
    if False:
        print('Hello World!')
    client = redis_v1.CloudRedisClient()
    request = redis_v1.UpgradeInstanceRequest(name='name_value', redis_version='redis_version_value')
    operation = client.upgrade_instance(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)