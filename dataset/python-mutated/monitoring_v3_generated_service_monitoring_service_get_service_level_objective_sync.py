from google.cloud import monitoring_v3

def sample_get_service_level_objective():
    if False:
        i = 10
        return i + 15
    client = monitoring_v3.ServiceMonitoringServiceClient()
    request = monitoring_v3.GetServiceLevelObjectiveRequest(name='name_value')
    response = client.get_service_level_objective(request=request)
    print(response)