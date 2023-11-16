from google.cloud import workstations_v1beta

def sample_delete_workstation_config():
    if False:
        for i in range(10):
            print('nop')
    client = workstations_v1beta.WorkstationsClient()
    request = workstations_v1beta.DeleteWorkstationConfigRequest(name='name_value')
    operation = client.delete_workstation_config(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)