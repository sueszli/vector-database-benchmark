from google.cloud.orchestration.airflow import service_v1

def sample_update_environment():
    if False:
        for i in range(10):
            print('nop')
    client = service_v1.EnvironmentsClient()
    request = service_v1.UpdateEnvironmentRequest()
    operation = client.update_environment(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)