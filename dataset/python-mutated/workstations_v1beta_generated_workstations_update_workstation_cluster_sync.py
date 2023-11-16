from google.cloud import workstations_v1beta

def sample_update_workstation_cluster():
    if False:
        return 10
    client = workstations_v1beta.WorkstationsClient()
    request = workstations_v1beta.UpdateWorkstationClusterRequest()
    operation = client.update_workstation_cluster(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)