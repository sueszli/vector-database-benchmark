from google.cloud.orchestration.airflow import service_v1beta1

def sample_restart_web_server():
    if False:
        return 10
    client = service_v1beta1.EnvironmentsClient()
    request = service_v1beta1.RestartWebServerRequest()
    operation = client.restart_web_server(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)