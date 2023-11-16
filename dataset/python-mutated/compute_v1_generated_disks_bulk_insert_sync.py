from google.cloud import compute_v1

def sample_bulk_insert():
    if False:
        print('Hello World!')
    client = compute_v1.DisksClient()
    request = compute_v1.BulkInsertDiskRequest(project='project_value', zone='zone_value')
    response = client.bulk_insert(request=request)
    print(response)