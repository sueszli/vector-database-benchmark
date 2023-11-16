from google.cloud import contentwarehouse_v1

def sample_search_documents():
    if False:
        return 10
    client = contentwarehouse_v1.DocumentServiceClient()
    request = contentwarehouse_v1.SearchDocumentsRequest(parent='parent_value')
    page_result = client.search_documents(request=request)
    for response in page_result:
        print(response)