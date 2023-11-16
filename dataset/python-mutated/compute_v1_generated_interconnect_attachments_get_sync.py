from google.cloud import compute_v1

def sample_get():
    if False:
        i = 10
        return i + 15
    client = compute_v1.InterconnectAttachmentsClient()
    request = compute_v1.GetInterconnectAttachmentRequest(interconnect_attachment='interconnect_attachment_value', project='project_value', region='region_value')
    response = client.get(request=request)
    print(response)