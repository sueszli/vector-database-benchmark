from google.cloud import compute_v1

def sample_stop_group_async_replication():
    if False:
        return 10
    client = compute_v1.DisksClient()
    request = compute_v1.StopGroupAsyncReplicationDiskRequest(project='project_value', zone='zone_value')
    response = client.stop_group_async_replication(request=request)
    print(response)