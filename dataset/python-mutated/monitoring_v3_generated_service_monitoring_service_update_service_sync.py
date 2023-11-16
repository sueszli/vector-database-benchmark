from google.cloud import monitoring_v3

def sample_update_service():
    if False:
        for i in range(10):
            print('nop')
    client = monitoring_v3.ServiceMonitoringServiceClient()
    request = monitoring_v3.UpdateServiceRequest()
    response = client.update_service(request=request)
    print(response)