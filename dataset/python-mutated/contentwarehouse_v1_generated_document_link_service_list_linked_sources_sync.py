from google.cloud import contentwarehouse_v1

def sample_list_linked_sources():
    if False:
        print('Hello World!')
    client = contentwarehouse_v1.DocumentLinkServiceClient()
    request = contentwarehouse_v1.ListLinkedSourcesRequest(parent='parent_value')
    page_result = client.list_linked_sources(request=request)
    for response in page_result:
        print(response)