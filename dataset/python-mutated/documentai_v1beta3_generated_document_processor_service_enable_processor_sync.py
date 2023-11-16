from google.cloud import documentai_v1beta3

def sample_enable_processor():
    if False:
        i = 10
        return i + 15
    client = documentai_v1beta3.DocumentProcessorServiceClient()
    request = documentai_v1beta3.EnableProcessorRequest(name='name_value')
    operation = client.enable_processor(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)