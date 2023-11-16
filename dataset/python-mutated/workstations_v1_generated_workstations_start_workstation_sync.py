from google.cloud import workstations_v1

def sample_start_workstation():
    if False:
        print('Hello World!')
    client = workstations_v1.WorkstationsClient()
    request = workstations_v1.StartWorkstationRequest(name='name_value')
    operation = client.start_workstation(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)