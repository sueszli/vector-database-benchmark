from google.cloud import documentai_v1

def sample_get_processor_type():
    if False:
        i = 10
        return i + 15
    client = documentai_v1.DocumentProcessorServiceClient()
    request = documentai_v1.GetProcessorTypeRequest(name='name_value')
    response = client.get_processor_type(request=request)
    print(response)