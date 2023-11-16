from google.cloud import container_v1beta1

def sample_set_logging_service():
    if False:
        while True:
            i = 10
    client = container_v1beta1.ClusterManagerClient()
    request = container_v1beta1.SetLoggingServiceRequest(project_id='project_id_value', zone='zone_value', cluster_id='cluster_id_value', logging_service='logging_service_value')
    response = client.set_logging_service(request=request)
    print(response)