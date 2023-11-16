from google.cloud import assuredworkloads_v1

def sample_get_workload():
    if False:
        for i in range(10):
            print('nop')
    client = assuredworkloads_v1.AssuredWorkloadsServiceClient()
    request = assuredworkloads_v1.GetWorkloadRequest(name='name_value')
    response = client.get_workload(request=request)
    print(response)