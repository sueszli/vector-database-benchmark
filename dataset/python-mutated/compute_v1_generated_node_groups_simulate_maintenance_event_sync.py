from google.cloud import compute_v1

def sample_simulate_maintenance_event():
    if False:
        for i in range(10):
            print('nop')
    client = compute_v1.NodeGroupsClient()
    request = compute_v1.SimulateMaintenanceEventNodeGroupRequest(node_group='node_group_value', project='project_value', zone='zone_value')
    response = client.simulate_maintenance_event(request=request)
    print(response)