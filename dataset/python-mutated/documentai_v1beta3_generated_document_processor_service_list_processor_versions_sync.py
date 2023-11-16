from google.cloud import documentai_v1beta3

def sample_list_processor_versions():
    if False:
        while True:
            i = 10
    client = documentai_v1beta3.DocumentProcessorServiceClient()
    request = documentai_v1beta3.ListProcessorVersionsRequest(parent='parent_value')
    page_result = client.list_processor_versions(request=request)
    for response in page_result:
        print(response)