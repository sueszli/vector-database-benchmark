from google.cloud import documentai_v1

def sample_train_processor_version():
    if False:
        while True:
            i = 10
    client = documentai_v1.DocumentProcessorServiceClient()
    request = documentai_v1.TrainProcessorVersionRequest(parent='parent_value')
    operation = client.train_processor_version(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)