from google.cloud import documentai_v1beta3

def sample_evaluate_processor_version():
    if False:
        while True:
            i = 10
    client = documentai_v1beta3.DocumentProcessorServiceClient()
    request = documentai_v1beta3.EvaluateProcessorVersionRequest(processor_version='processor_version_value')
    operation = client.evaluate_processor_version(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)