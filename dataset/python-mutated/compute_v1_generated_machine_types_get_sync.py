from google.cloud import compute_v1

def sample_get():
    if False:
        print('Hello World!')
    client = compute_v1.MachineTypesClient()
    request = compute_v1.GetMachineTypeRequest(machine_type='machine_type_value', project='project_value', zone='zone_value')
    response = client.get(request=request)
    print(response)