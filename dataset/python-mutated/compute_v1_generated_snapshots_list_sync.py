from google.cloud import compute_v1

def sample_list():
    if False:
        i = 10
        return i + 15
    client = compute_v1.SnapshotsClient()
    request = compute_v1.ListSnapshotsRequest(project='project_value')
    page_result = client.list(request=request)
    for response in page_result:
        print(response)