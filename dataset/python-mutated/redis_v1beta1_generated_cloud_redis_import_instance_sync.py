from google.cloud import redis_v1beta1

def sample_import_instance():
    if False:
        return 10
    client = redis_v1beta1.CloudRedisClient()
    input_config = redis_v1beta1.InputConfig()
    input_config.gcs_source.uri = 'uri_value'
    request = redis_v1beta1.ImportInstanceRequest(name='name_value', input_config=input_config)
    operation = client.import_instance(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)