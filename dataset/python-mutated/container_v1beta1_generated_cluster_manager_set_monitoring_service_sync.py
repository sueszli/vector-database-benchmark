from google.cloud import container_v1beta1

def sample_set_monitoring_service():
    if False:
        return 10
    client = container_v1beta1.ClusterManagerClient()
    request = container_v1beta1.SetMonitoringServiceRequest(project_id='project_id_value', zone='zone_value', cluster_id='cluster_id_value', monitoring_service='monitoring_service_value')
    response = client.set_monitoring_service(request=request)
    print(response)