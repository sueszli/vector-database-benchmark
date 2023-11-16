from google.cloud import compute_v1

def sample_insert():
    if False:
        print('Hello World!')
    client = compute_v1.NetworkAttachmentsClient()
    request = compute_v1.InsertNetworkAttachmentRequest(project='project_value', region='region_value')
    response = client.insert(request=request)
    print(response)