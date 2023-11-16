from google.cloud import workstations_v1

def sample_create_workstation():
    if False:
        print('Hello World!')
    client = workstations_v1.WorkstationsClient()
    request = workstations_v1.CreateWorkstationRequest(parent='parent_value', workstation_id='workstation_id_value')
    operation = client.create_workstation(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)