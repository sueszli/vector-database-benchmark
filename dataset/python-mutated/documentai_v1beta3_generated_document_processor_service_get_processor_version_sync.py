from google.cloud import documentai_v1beta3

def sample_get_processor_version():
    if False:
        while True:
            i = 10
    client = documentai_v1beta3.DocumentProcessorServiceClient()
    request = documentai_v1beta3.GetProcessorVersionRequest(name='name_value')
    response = client.get_processor_version(request=request)
    print(response)