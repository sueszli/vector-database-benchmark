from google.cloud import edgenetwork_v1

def sample_get_interconnect_attachment():
    if False:
        return 10
    client = edgenetwork_v1.EdgeNetworkClient()
    request = edgenetwork_v1.GetInterconnectAttachmentRequest(name='name_value')
    response = client.get_interconnect_attachment(request=request)
    print(response)