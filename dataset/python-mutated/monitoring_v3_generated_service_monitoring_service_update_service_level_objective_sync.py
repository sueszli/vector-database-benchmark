from google.cloud import monitoring_v3

def sample_update_service_level_objective():
    if False:
        while True:
            i = 10
    client = monitoring_v3.ServiceMonitoringServiceClient()
    request = monitoring_v3.UpdateServiceLevelObjectiveRequest()
    response = client.update_service_level_objective(request=request)
    print(response)