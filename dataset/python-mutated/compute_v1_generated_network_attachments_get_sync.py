from google.cloud import compute_v1

def sample_get():
    if False:
        for i in range(10):
            print('nop')
    client = compute_v1.NetworkAttachmentsClient()
    request = compute_v1.GetNetworkAttachmentRequest(network_attachment='network_attachment_value', project='project_value', region='region_value')
    response = client.get(request=request)
    print(response)