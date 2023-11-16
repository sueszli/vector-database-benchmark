from google.cloud import compute_v1

def sample_insert():
    if False:
        while True:
            i = 10
    client = compute_v1.SnapshotsClient()
    request = compute_v1.InsertSnapshotRequest(project='project_value')
    response = client.insert(request=request)
    print(response)