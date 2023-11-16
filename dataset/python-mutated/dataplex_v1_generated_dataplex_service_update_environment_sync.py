from google.cloud import dataplex_v1

def sample_update_environment():
    if False:
        while True:
            i = 10
    client = dataplex_v1.DataplexServiceClient()
    environment = dataplex_v1.Environment()
    environment.infrastructure_spec.os_image.image_version = 'image_version_value'
    request = dataplex_v1.UpdateEnvironmentRequest(environment=environment)
    operation = client.update_environment(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)