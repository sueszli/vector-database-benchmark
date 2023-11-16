from google.cloud import documentai_v1beta3

def sample_list_evaluations():
    if False:
        print('Hello World!')
    client = documentai_v1beta3.DocumentProcessorServiceClient()
    request = documentai_v1beta3.ListEvaluationsRequest(parent='parent_value')
    page_result = client.list_evaluations(request=request)
    for response in page_result:
        print(response)