from google.cloud import compute_v1

def sample_set_labels():
    if False:
        for i in range(10):
            print('nop')
    client = compute_v1.SnapshotsClient()
    request = compute_v1.SetLabelsSnapshotRequest(project='project_value', resource='resource_value')
    response = client.set_labels(request=request)
    print(response)