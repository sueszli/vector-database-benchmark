from google.cloud import documentai_v1

def sample_set_default_processor_version():
    if False:
        while True:
            i = 10
    client = documentai_v1.DocumentProcessorServiceClient()
    request = documentai_v1.SetDefaultProcessorVersionRequest(processor='processor_value', default_processor_version='default_processor_version_value')
    operation = client.set_default_processor_version(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)