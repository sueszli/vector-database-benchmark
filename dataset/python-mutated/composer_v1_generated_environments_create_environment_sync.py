from google.cloud.orchestration.airflow import service_v1

def sample_create_environment():
    if False:
        return 10
    client = service_v1.EnvironmentsClient()
    request = service_v1.CreateEnvironmentRequest()
    operation = client.create_environment(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)