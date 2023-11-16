from google.cloud.orchestration.airflow import service_v1

def sample_delete_environment():
    if False:
        i = 10
        return i + 15
    client = service_v1.EnvironmentsClient()
    request = service_v1.DeleteEnvironmentRequest()
    operation = client.delete_environment(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)