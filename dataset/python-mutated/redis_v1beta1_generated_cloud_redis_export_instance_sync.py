from google.cloud import redis_v1beta1

def sample_export_instance():
    if False:
        for i in range(10):
            print('nop')
    client = redis_v1beta1.CloudRedisClient()
    output_config = redis_v1beta1.OutputConfig()
    output_config.gcs_destination.uri = 'uri_value'
    request = redis_v1beta1.ExportInstanceRequest(name='name_value', output_config=output_config)
    operation = client.export_instance(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)