from google.cloud import assuredworkloads_v1beta1

def sample_get_workload():
    if False:
        return 10
    client = assuredworkloads_v1beta1.AssuredWorkloadsServiceClient()
    request = assuredworkloads_v1beta1.GetWorkloadRequest(name='name_value')
    response = client.get_workload(request=request)
    print(response)