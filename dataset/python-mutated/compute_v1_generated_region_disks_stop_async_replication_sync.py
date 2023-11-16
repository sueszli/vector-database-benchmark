from google.cloud import compute_v1

def sample_stop_async_replication():
    if False:
        print('Hello World!')
    client = compute_v1.RegionDisksClient()
    request = compute_v1.StopAsyncReplicationRegionDiskRequest(disk='disk_value', project='project_value', region='region_value')
    response = client.stop_async_replication(request=request)
    print(response)