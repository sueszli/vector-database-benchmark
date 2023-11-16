from google.cloud import workstations_v1

def sample_update_workstation_config():
    if False:
        while True:
            i = 10
    client = workstations_v1.WorkstationsClient()
    request = workstations_v1.UpdateWorkstationConfigRequest()
    operation = client.update_workstation_config(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)