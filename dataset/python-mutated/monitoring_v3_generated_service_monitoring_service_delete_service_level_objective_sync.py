from google.cloud import monitoring_v3

def sample_delete_service_level_objective():
    if False:
        while True:
            i = 10
    client = monitoring_v3.ServiceMonitoringServiceClient()
    request = monitoring_v3.DeleteServiceLevelObjectiveRequest(name='name_value')
    client.delete_service_level_objective(request=request)