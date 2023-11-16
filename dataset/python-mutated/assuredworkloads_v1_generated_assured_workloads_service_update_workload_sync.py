from google.cloud import assuredworkloads_v1

def sample_update_workload():
    if False:
        for i in range(10):
            print('nop')
    client = assuredworkloads_v1.AssuredWorkloadsServiceClient()
    workload = assuredworkloads_v1.Workload()
    workload.display_name = 'display_name_value'
    workload.compliance_regime = 'ASSURED_WORKLOADS_FOR_PARTNERS'
    request = assuredworkloads_v1.UpdateWorkloadRequest(workload=workload)
    response = client.update_workload(request=request)
    print(response)