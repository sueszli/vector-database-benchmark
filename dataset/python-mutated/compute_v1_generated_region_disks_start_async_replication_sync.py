from google.cloud import compute_v1

def sample_start_async_replication():
    if False:
        i = 10
        return i + 15
    client = compute_v1.RegionDisksClient()
    request = compute_v1.StartAsyncReplicationRegionDiskRequest(disk='disk_value', project='project_value', region='region_value')
    response = client.start_async_replication(request=request)
    print(response)