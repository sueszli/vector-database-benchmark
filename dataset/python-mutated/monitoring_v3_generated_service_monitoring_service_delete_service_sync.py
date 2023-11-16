from google.cloud import monitoring_v3

def sample_delete_service():
    if False:
        for i in range(10):
            print('nop')
    client = monitoring_v3.ServiceMonitoringServiceClient()
    request = monitoring_v3.DeleteServiceRequest(name='name_value')
    client.delete_service(request=request)