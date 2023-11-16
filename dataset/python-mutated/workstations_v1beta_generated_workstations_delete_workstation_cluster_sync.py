from google.cloud import workstations_v1beta

def sample_delete_workstation_cluster():
    if False:
        return 10
    client = workstations_v1beta.WorkstationsClient()
    request = workstations_v1beta.DeleteWorkstationClusterRequest(name='name_value')
    operation = client.delete_workstation_cluster(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)