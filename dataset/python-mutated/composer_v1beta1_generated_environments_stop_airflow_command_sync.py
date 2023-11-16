from google.cloud.orchestration.airflow import service_v1beta1

def sample_stop_airflow_command():
    if False:
        print('Hello World!')
    client = service_v1beta1.EnvironmentsClient()
    request = service_v1beta1.StopAirflowCommandRequest()
    response = client.stop_airflow_command(request=request)
    print(response)