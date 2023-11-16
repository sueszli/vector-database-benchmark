from google.cloud import monitoring_v3

def sample_create_group():
    if False:
        return 10
    client = monitoring_v3.GroupServiceClient()
    request = monitoring_v3.CreateGroupRequest(name='name_value')
    response = client.create_group(request=request)
    print(response)