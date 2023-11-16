from google.cloud import monitoring_v3

def sample_create_service_level_objective():
    if False:
        i = 10
        return i + 15
    client = monitoring_v3.ServiceMonitoringServiceClient()
    request = monitoring_v3.CreateServiceLevelObjectiveRequest(parent='parent_value')
    response = client.create_service_level_objective(request=request)
    print(response)