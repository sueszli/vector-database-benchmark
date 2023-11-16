from google.cloud import compute_v1

def sample_bulk_insert():
    if False:
        while True:
            i = 10
    client = compute_v1.InstancesClient()
    request = compute_v1.BulkInsertInstanceRequest(project='project_value', zone='zone_value')
    response = client.bulk_insert(request=request)
    print(response)