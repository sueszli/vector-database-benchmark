from google.cloud import compute_v1

def sample_delete():
    if False:
        while True:
            i = 10
    client = compute_v1.ServiceAttachmentsClient()
    request = compute_v1.DeleteServiceAttachmentRequest(project='project_value', region='region_value', service_attachment='service_attachment_value')
    response = client.delete(request=request)
    print(response)