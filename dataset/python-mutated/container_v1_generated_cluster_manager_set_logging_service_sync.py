from google.cloud import container_v1

def sample_set_logging_service():
    if False:
        print('Hello World!')
    client = container_v1.ClusterManagerClient()
    request = container_v1.SetLoggingServiceRequest(logging_service='logging_service_value')
    response = client.set_logging_service(request=request)
    print(response)