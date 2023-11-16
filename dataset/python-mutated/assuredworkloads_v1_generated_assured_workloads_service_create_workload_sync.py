from google.cloud import assuredworkloads_v1

def sample_create_workload():
    if False:
        print('Hello World!')
    client = assuredworkloads_v1.AssuredWorkloadsServiceClient()
    workload = assuredworkloads_v1.Workload()
    workload.display_name = 'display_name_value'
    workload.compliance_regime = 'ASSURED_WORKLOADS_FOR_PARTNERS'
    request = assuredworkloads_v1.CreateWorkloadRequest(parent='parent_value', workload=workload)
    operation = client.create_workload(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)