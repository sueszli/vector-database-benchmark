from google.cloud import assuredworkloads_v1beta1

def sample_delete_workload():
    if False:
        print('Hello World!')
    client = assuredworkloads_v1beta1.AssuredWorkloadsServiceClient()
    request = assuredworkloads_v1beta1.DeleteWorkloadRequest(name='name_value')
    client.delete_workload(request=request)