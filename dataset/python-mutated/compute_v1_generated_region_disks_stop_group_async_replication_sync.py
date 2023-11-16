from google.cloud import compute_v1

def sample_stop_group_async_replication():
    if False:
        for i in range(10):
            print('nop')
    client = compute_v1.RegionDisksClient()
    request = compute_v1.StopGroupAsyncReplicationRegionDiskRequest(project='project_value', region='region_value')
    response = client.stop_group_async_replication(request=request)
    print(response)