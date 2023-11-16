from google.cloud import workstations_v1

def sample_update_workstation_cluster():
    if False:
        while True:
            i = 10
    client = workstations_v1.WorkstationsClient()
    request = workstations_v1.UpdateWorkstationClusterRequest()
    operation = client.update_workstation_cluster(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)