from google.cloud import documentai_v1beta3

def sample_list_documents():
    if False:
        for i in range(10):
            print('nop')
    client = documentai_v1beta3.DocumentServiceClient()
    request = documentai_v1beta3.ListDocumentsRequest(dataset='dataset_value')
    page_result = client.list_documents(request=request)
    for response in page_result:
        print(response)