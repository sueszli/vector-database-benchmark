from google.cloud import compute_v1

def sample_simulate_maintenance_event():
    if False:
        while True:
            i = 10
    client = compute_v1.InstancesClient()
    request = compute_v1.SimulateMaintenanceEventInstanceRequest(instance='instance_value', project='project_value', zone='zone_value')
    response = client.simulate_maintenance_event(request=request)
    print(response)