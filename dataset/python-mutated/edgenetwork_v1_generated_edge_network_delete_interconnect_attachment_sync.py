from google.cloud import edgenetwork_v1

def sample_delete_interconnect_attachment():
    if False:
        print('Hello World!')
    client = edgenetwork_v1.EdgeNetworkClient()
    request = edgenetwork_v1.DeleteInterconnectAttachmentRequest(name='name_value')
    operation = client.delete_interconnect_attachment(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)