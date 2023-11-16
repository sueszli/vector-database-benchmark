from google.cloud import contentwarehouse_v1

def sample_list_linked_targets():
    if False:
        i = 10
        return i + 15
    client = contentwarehouse_v1.DocumentLinkServiceClient()
    request = contentwarehouse_v1.ListLinkedTargetsRequest(parent='parent_value')
    response = client.list_linked_targets(request=request)
    print(response)