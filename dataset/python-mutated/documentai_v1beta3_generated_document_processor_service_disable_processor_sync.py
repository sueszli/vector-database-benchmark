from google.cloud import documentai_v1beta3

def sample_disable_processor():
    if False:
        return 10
    client = documentai_v1beta3.DocumentProcessorServiceClient()
    request = documentai_v1beta3.DisableProcessorRequest(name='name_value')
    operation = client.disable_processor(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)