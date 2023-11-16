from google.cloud import compute_v1

def sample_patch():
    if False:
        return 10
    client = compute_v1.InterconnectAttachmentsClient()
    request = compute_v1.PatchInterconnectAttachmentRequest(interconnect_attachment='interconnect_attachment_value', project='project_value', region='region_value')
    response = client.patch(request=request)
    print(response)