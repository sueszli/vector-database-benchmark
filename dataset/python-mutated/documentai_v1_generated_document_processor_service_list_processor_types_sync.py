from google.cloud import documentai_v1

def sample_list_processor_types():
    if False:
        while True:
            i = 10
    client = documentai_v1.DocumentProcessorServiceClient()
    request = documentai_v1.ListProcessorTypesRequest(parent='parent_value')
    page_result = client.list_processor_types(request=request)
    for response in page_result:
        print(response)