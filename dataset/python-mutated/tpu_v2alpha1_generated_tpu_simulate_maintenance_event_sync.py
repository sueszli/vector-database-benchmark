from google.cloud import tpu_v2alpha1

def sample_simulate_maintenance_event():
    if False:
        for i in range(10):
            print('nop')
    client = tpu_v2alpha1.TpuClient()
    request = tpu_v2alpha1.SimulateMaintenanceEventRequest(name='name_value')
    operation = client.simulate_maintenance_event(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)