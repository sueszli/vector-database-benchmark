from google.cloud import assuredworkloads_v1

def sample_delete_workload():
    if False:
        i = 10
        return i + 15
    client = assuredworkloads_v1.AssuredWorkloadsServiceClient()
    request = assuredworkloads_v1.DeleteWorkloadRequest(name='name_value')
    client.delete_workload(request=request)