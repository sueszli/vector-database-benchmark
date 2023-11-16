from google.cloud import documentai_v1

def sample_list_evaluations():
    if False:
        i = 10
        return i + 15
    client = documentai_v1.DocumentProcessorServiceClient()
    request = documentai_v1.ListEvaluationsRequest(parent='parent_value')
    page_result = client.list_evaluations(request=request)
    for response in page_result:
        print(response)