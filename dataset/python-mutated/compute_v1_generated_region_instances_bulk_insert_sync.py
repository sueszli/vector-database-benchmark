from google.cloud import compute_v1

def sample_bulk_insert():
    if False:
        return 10
    client = compute_v1.RegionInstancesClient()
    request = compute_v1.BulkInsertRegionInstanceRequest(project='project_value', region='region_value')
    response = client.bulk_insert(request=request)
    print(response)