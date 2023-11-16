from google.cloud import documentai_v1beta3

def sample_batch_process_documents():
    if False:
        while True:
            i = 10
    client = documentai_v1beta3.DocumentProcessorServiceClient()
    request = documentai_v1beta3.BatchProcessRequest(name='name_value')
    operation = client.batch_process_documents(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)