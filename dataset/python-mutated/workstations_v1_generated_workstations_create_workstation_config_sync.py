from google.cloud import workstations_v1

def sample_create_workstation_config():
    if False:
        return 10
    client = workstations_v1.WorkstationsClient()
    request = workstations_v1.CreateWorkstationConfigRequest(parent='parent_value', workstation_config_id='workstation_config_id_value')
    operation = client.create_workstation_config(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)