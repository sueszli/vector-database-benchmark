from google.cloud import container_v1

def sample_set_maintenance_policy():
    if False:
        return 10
    client = container_v1.ClusterManagerClient()
    request = container_v1.SetMaintenancePolicyRequest(project_id='project_id_value', zone='zone_value', cluster_id='cluster_id_value')
    response = client.set_maintenance_policy(request=request)
    print(response)