from google.cloud.orchestration.airflow import service_v1beta1

def sample_delete_environment():
    if False:
        print('Hello World!')
    client = service_v1beta1.EnvironmentsClient()
    request = service_v1beta1.DeleteEnvironmentRequest()
    operation = client.delete_environment(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)