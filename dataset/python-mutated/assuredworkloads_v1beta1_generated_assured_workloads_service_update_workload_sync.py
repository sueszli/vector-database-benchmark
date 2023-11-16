from google.cloud import assuredworkloads_v1beta1

def sample_update_workload():
    if False:
        return 10
    client = assuredworkloads_v1beta1.AssuredWorkloadsServiceClient()
    workload = assuredworkloads_v1beta1.Workload()
    workload.display_name = 'display_name_value'
    workload.compliance_regime = 'AU_REGIONS_AND_US_SUPPORT'
    request = assuredworkloads_v1beta1.UpdateWorkloadRequest(workload=workload)
    response = client.update_workload(request=request)
    print(response)