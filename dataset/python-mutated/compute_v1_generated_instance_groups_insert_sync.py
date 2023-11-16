from google.cloud import compute_v1

def sample_insert():
    if False:
        for i in range(10):
            print('nop')
    client = compute_v1.InstanceGroupsClient()
    request = compute_v1.InsertInstanceGroupRequest(project='project_value', zone='zone_value')
    response = client.insert(request=request)
    print(response)