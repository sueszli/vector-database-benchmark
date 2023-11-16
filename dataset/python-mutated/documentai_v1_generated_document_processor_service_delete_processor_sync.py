from google.cloud import documentai_v1

def sample_delete_processor():
    if False:
        while True:
            i = 10
    client = documentai_v1.DocumentProcessorServiceClient()
    request = documentai_v1.DeleteProcessorRequest(name='name_value')
    operation = client.delete_processor(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)