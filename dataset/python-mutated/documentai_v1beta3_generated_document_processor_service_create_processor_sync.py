from google.cloud import documentai_v1beta3

def sample_create_processor():
    if False:
        while True:
            i = 10
    client = documentai_v1beta3.DocumentProcessorServiceClient()
    request = documentai_v1beta3.CreateProcessorRequest(parent='parent_value')
    response = client.create_processor(request=request)
    print(response)