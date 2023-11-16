from google.cloud import container_v1beta1

def sample_set_locations():
    if False:
        i = 10
        return i + 15
    client = container_v1beta1.ClusterManagerClient()
    request = container_v1beta1.SetLocationsRequest(project_id='project_id_value', zone='zone_value', cluster_id='cluster_id_value', locations=['locations_value1', 'locations_value2'])
    response = client.set_locations(request=request)
    print(response)