from google.cloud import documentai_v1

def sample_undeploy_processor_version():
    if False:
        print('Hello World!')
    client = documentai_v1.DocumentProcessorServiceClient()
    request = documentai_v1.UndeployProcessorVersionRequest(name='name_value')
    operation = client.undeploy_processor_version(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)