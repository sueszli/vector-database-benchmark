from google.cloud import compute_v1

def sample_patch():
    if False:
        i = 10
        return i + 15
    client = compute_v1.ServiceAttachmentsClient()
    request = compute_v1.PatchServiceAttachmentRequest(project='project_value', region='region_value', service_attachment='service_attachment_value')
    response = client.patch(request=request)
    print(response)