from google.cloud import workstations_v1

def sample_delete_workstation_config():
    if False:
        return 10
    client = workstations_v1.WorkstationsClient()
    request = workstations_v1.DeleteWorkstationConfigRequest(name='name_value')
    operation = client.delete_workstation_config(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)