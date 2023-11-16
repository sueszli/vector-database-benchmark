from google.cloud import workstations_v1

def sample_create_workstation_cluster():
    if False:
        return 10
    client = workstations_v1.WorkstationsClient()
    request = workstations_v1.CreateWorkstationClusterRequest(parent='parent_value', workstation_cluster_id='workstation_cluster_id_value')
    operation = client.create_workstation_cluster(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)