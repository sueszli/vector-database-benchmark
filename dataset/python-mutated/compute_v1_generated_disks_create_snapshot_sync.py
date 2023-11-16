from google.cloud import compute_v1

def sample_create_snapshot():
    if False:
        return 10
    client = compute_v1.DisksClient()
    request = compute_v1.CreateSnapshotDiskRequest(disk='disk_value', project='project_value', zone='zone_value')
    response = client.create_snapshot(request=request)
    print(response)