from google.cloud import documentai_v1

def sample_enable_processor():
    if False:
        return 10
    client = documentai_v1.DocumentProcessorServiceClient()
    request = documentai_v1.EnableProcessorRequest(name='name_value')
    operation = client.enable_processor(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)