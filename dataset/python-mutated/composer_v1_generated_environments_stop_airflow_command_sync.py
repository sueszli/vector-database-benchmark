from google.cloud.orchestration.airflow import service_v1

def sample_stop_airflow_command():
    if False:
        for i in range(10):
            print('nop')
    client = service_v1.EnvironmentsClient()
    request = service_v1.StopAirflowCommandRequest()
    response = client.stop_airflow_command(request=request)
    print(response)