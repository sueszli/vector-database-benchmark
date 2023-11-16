from google.cloud import monitoring_v3

def sample_delete_group():
    if False:
        print('Hello World!')
    client = monitoring_v3.GroupServiceClient()
    request = monitoring_v3.DeleteGroupRequest(name='name_value')
    client.delete_group(request=request)