from google.cloud import compute_v1

def sample_insert():
    if False:
        print('Hello World!')
    client = compute_v1.InstancesClient()
    request = compute_v1.InsertInstanceRequest(project='project_value', zone='zone_value')
    response = client.insert(request=request)
    print(response)