from google.cloud import compute_v1

def sample_list():
    if False:
        return 10
    client = compute_v1.InterconnectAttachmentsClient()
    request = compute_v1.ListInterconnectAttachmentsRequest(project='project_value', region='region_value')
    page_result = client.list(request=request)
    for response in page_result:
        print(response)