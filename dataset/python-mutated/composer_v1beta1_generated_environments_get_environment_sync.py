from google.cloud.orchestration.airflow import service_v1beta1

def sample_get_environment():
    if False:
        for i in range(10):
            print('nop')
    client = service_v1beta1.EnvironmentsClient()
    request = service_v1beta1.GetEnvironmentRequest()
    response = client.get_environment(request=request)
    print(response)