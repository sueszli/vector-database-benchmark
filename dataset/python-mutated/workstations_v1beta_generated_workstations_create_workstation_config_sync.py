from google.cloud import workstations_v1beta

def sample_create_workstation_config():
    if False:
        print('Hello World!')
    client = workstations_v1beta.WorkstationsClient()
    request = workstations_v1beta.CreateWorkstationConfigRequest(parent='parent_value', workstation_config_id='workstation_config_id_value')
    operation = client.create_workstation_config(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)