from google.cloud import documentai_v1beta3

def sample_import_processor_version():
    if False:
        while True:
            i = 10
    client = documentai_v1beta3.DocumentProcessorServiceClient()
    request = documentai_v1beta3.ImportProcessorVersionRequest(processor_version_source='processor_version_source_value', parent='parent_value')
    operation = client.import_processor_version(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)