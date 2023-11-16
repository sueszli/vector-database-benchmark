from google.cloud import documentai_v1

def sample_batch_process_documents():
    if False:
        i = 10
        return i + 15
    client = documentai_v1.DocumentProcessorServiceClient()
    request = documentai_v1.BatchProcessRequest(name='name_value')
    operation = client.batch_process_documents(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)