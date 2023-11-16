from google.cloud import compute_v1

def sample_bulk_insert():
    if False:
        return 10
    client = compute_v1.RegionDisksClient()
    request = compute_v1.BulkInsertRegionDiskRequest(project='project_value', region='region_value')
    response = client.bulk_insert(request=request)
    print(response)