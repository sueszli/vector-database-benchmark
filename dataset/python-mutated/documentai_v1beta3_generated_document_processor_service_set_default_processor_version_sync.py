from google.cloud import documentai_v1beta3

def sample_set_default_processor_version():
    if False:
        for i in range(10):
            print('nop')
    client = documentai_v1beta3.DocumentProcessorServiceClient()
    request = documentai_v1beta3.SetDefaultProcessorVersionRequest(processor='processor_value', default_processor_version='default_processor_version_value')
    operation = client.set_default_processor_version(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)