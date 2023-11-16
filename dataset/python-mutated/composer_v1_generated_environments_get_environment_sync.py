from google.cloud.orchestration.airflow import service_v1

def sample_get_environment():
    if False:
        i = 10
        return i + 15
    client = service_v1.EnvironmentsClient()
    request = service_v1.GetEnvironmentRequest()
    response = client.get_environment(request=request)
    print(response)