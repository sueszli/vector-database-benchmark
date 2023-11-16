from google.cloud import documentai_v1

def sample_fetch_processor_types():
    if False:
        return 10
    client = documentai_v1.DocumentProcessorServiceClient()
    request = documentai_v1.FetchProcessorTypesRequest(parent='parent_value')
    response = client.fetch_processor_types(request=request)
    print(response)