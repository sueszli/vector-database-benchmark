from google.cloud import compute_v1

def sample_delete():
    if False:
        for i in range(10):
            print('nop')
    client = compute_v1.SnapshotsClient()
    request = compute_v1.DeleteSnapshotRequest(project='project_value', snapshot='snapshot_value')
    response = client.delete(request=request)
    print(response)