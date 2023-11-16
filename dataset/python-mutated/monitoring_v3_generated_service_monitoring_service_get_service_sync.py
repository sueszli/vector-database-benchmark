from google.cloud import monitoring_v3

def sample_get_service():
    if False:
        while True:
            i = 10
    client = monitoring_v3.ServiceMonitoringServiceClient()
    request = monitoring_v3.GetServiceRequest(name='name_value')
    response = client.get_service(request=request)
    print(response)