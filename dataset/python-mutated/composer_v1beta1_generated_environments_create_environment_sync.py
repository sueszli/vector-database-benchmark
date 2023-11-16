from google.cloud.orchestration.airflow import service_v1beta1

def sample_create_environment():
    if False:
        for i in range(10):
            print('nop')
    client = service_v1beta1.EnvironmentsClient()
    request = service_v1beta1.CreateEnvironmentRequest()
    operation = client.create_environment(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)