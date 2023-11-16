from google.cloud import edgenetwork_v1

def sample_list_interconnect_attachments():
    if False:
        print('Hello World!')
    client = edgenetwork_v1.EdgeNetworkClient()
    request = edgenetwork_v1.ListInterconnectAttachmentsRequest(parent='parent_value')
    page_result = client.list_interconnect_attachments(request=request)
    for response in page_result:
        print(response)