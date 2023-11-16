from google.cloud import documentai_v1

def sample_get_processor():
    if False:
        print('Hello World!')
    client = documentai_v1.DocumentProcessorServiceClient()
    request = documentai_v1.GetProcessorRequest(name='name_value')
    response = client.get_processor(request=request)
    print(response)