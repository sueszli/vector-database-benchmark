from google.cloud import documentai_v1beta3

def sample_undeploy_processor_version():
    if False:
        while True:
            i = 10
    client = documentai_v1beta3.DocumentProcessorServiceClient()
    request = documentai_v1beta3.UndeployProcessorVersionRequest(name='name_value')
    operation = client.undeploy_processor_version(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)