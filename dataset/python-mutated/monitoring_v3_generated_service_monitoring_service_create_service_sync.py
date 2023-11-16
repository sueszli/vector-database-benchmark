from google.cloud import monitoring_v3

def sample_create_service():
    if False:
        for i in range(10):
            print('nop')
    client = monitoring_v3.ServiceMonitoringServiceClient()
    request = monitoring_v3.CreateServiceRequest(parent='parent_value')
    response = client.create_service(request=request)
    print(response)