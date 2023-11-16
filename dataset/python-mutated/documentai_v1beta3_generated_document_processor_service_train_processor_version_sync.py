from google.cloud import documentai_v1beta3

def sample_train_processor_version():
    if False:
        return 10
    client = documentai_v1beta3.DocumentProcessorServiceClient()
    request = documentai_v1beta3.TrainProcessorVersionRequest(parent='parent_value')
    operation = client.train_processor_version(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)