from google.cloud import documentai_v1

def sample_list_processor_versions():
    if False:
        return 10
    client = documentai_v1.DocumentProcessorServiceClient()
    request = documentai_v1.ListProcessorVersionsRequest(parent='parent_value')
    page_result = client.list_processor_versions(request=request)
    for response in page_result:
        print(response)