from google.cloud import container_v1

def sample_set_monitoring_service():
    if False:
        while True:
            i = 10
    client = container_v1.ClusterManagerClient()
    request = container_v1.SetMonitoringServiceRequest(monitoring_service='monitoring_service_value')
    response = client.set_monitoring_service(request=request)
    print(response)