from google.cloud import compute_v1

def sample_get():
    if False:
        print('Hello World!')
    client = compute_v1.ServiceAttachmentsClient()
    request = compute_v1.GetServiceAttachmentRequest(project='project_value', region='region_value', service_attachment='service_attachment_value')
    response = client.get(request=request)
    print(response)