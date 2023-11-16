from google.cloud import edgenetwork_v1

def sample_create_interconnect_attachment():
    if False:
        for i in range(10):
            print('nop')
    client = edgenetwork_v1.EdgeNetworkClient()
    interconnect_attachment = edgenetwork_v1.InterconnectAttachment()
    interconnect_attachment.name = 'name_value'
    interconnect_attachment.interconnect = 'interconnect_value'
    interconnect_attachment.vlan_id = 733
    request = edgenetwork_v1.CreateInterconnectAttachmentRequest(parent='parent_value', interconnect_attachment_id='interconnect_attachment_id_value', interconnect_attachment=interconnect_attachment)
    operation = client.create_interconnect_attachment(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)