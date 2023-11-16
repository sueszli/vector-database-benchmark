from google.cloud import compute_v1

def sample_insert():
    if False:
        for i in range(10):
            print('nop')
    client = compute_v1.DisksClient()
    request = compute_v1.InsertDiskRequest(project='project_value', zone='zone_value')
    response = client.insert(request=request)
    print(response)