from google.cloud import documentai_v1beta3

def sample_fetch_processor_types():
    if False:
        i = 10
        return i + 15
    client = documentai_v1beta3.DocumentProcessorServiceClient()
    request = documentai_v1beta3.FetchProcessorTypesRequest(parent='parent_value')
    response = client.fetch_processor_types(request=request)
    print(response)