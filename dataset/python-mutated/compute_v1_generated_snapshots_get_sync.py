from google.cloud import compute_v1

def sample_get():
    if False:
        while True:
            i = 10
    client = compute_v1.SnapshotsClient()
    request = compute_v1.GetSnapshotRequest(project='project_value', snapshot='snapshot_value')
    response = client.get(request=request)
    print(response)