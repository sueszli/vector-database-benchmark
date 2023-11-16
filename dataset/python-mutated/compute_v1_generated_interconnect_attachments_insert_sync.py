from google.cloud import compute_v1

def sample_insert():
    if False:
        i = 10
        return i + 15
    client = compute_v1.InterconnectAttachmentsClient()
    request = compute_v1.InsertInterconnectAttachmentRequest(project='project_value', region='region_value')
    response = client.insert(request=request)
    print(response)