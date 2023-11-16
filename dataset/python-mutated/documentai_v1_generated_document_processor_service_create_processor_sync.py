from google.cloud import documentai_v1

def sample_create_processor():
    if False:
        return 10
    client = documentai_v1.DocumentProcessorServiceClient()
    request = documentai_v1.CreateProcessorRequest(parent='parent_value')
    response = client.create_processor(request=request)
    print(response)