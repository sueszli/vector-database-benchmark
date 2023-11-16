from google.cloud.orchestration.airflow import service_v1beta1

def sample_update_environment():
    if False:
        while True:
            i = 10
    client = service_v1beta1.EnvironmentsClient()
    request = service_v1beta1.UpdateEnvironmentRequest()
    operation = client.update_environment(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)