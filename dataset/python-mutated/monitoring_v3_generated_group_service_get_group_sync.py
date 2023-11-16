from google.cloud import monitoring_v3

def sample_get_group():
    if False:
        for i in range(10):
            print('nop')
    client = monitoring_v3.GroupServiceClient()
    request = monitoring_v3.GetGroupRequest(name='name_value')
    response = client.get_group(request=request)
    print(response)