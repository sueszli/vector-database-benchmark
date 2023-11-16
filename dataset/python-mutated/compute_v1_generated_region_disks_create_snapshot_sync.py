from google.cloud import compute_v1

def sample_create_snapshot():
    if False:
        i = 10
        return i + 15
    client = compute_v1.RegionDisksClient()
    request = compute_v1.CreateSnapshotRegionDiskRequest(disk='disk_value', project='project_value', region='region_value')
    response = client.create_snapshot(request=request)
    print(response)