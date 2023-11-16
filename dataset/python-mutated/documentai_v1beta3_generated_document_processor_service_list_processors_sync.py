from google.cloud import documentai_v1beta3

def sample_list_processors():
    if False:
        i = 10
        return i + 15
    client = documentai_v1beta3.DocumentProcessorServiceClient()
    request = documentai_v1beta3.ListProcessorsRequest(parent='parent_value')
    page_result = client.list_processors(request=request)
    for response in page_result:
        print(response)