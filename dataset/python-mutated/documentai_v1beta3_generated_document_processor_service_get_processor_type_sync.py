from google.cloud import documentai_v1beta3

def sample_get_processor_type():
    if False:
        print('Hello World!')
    client = documentai_v1beta3.DocumentProcessorServiceClient()
    request = documentai_v1beta3.GetProcessorTypeRequest(name='name_value')
    response = client.get_processor_type(request=request)
    print(response)