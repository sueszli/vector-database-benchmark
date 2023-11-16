from google.cloud import monitoring_v3

def sample_update_group():
    if False:
        for i in range(10):
            print('nop')
    client = monitoring_v3.GroupServiceClient()
    request = monitoring_v3.UpdateGroupRequest()
    response = client.update_group(request=request)
    print(response)