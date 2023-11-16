from google.cloud.orchestration.airflow import service_v1beta1

def sample_execute_airflow_command():
    if False:
        while True:
            i = 10
    client = service_v1beta1.EnvironmentsClient()
    request = service_v1beta1.ExecuteAirflowCommandRequest()
    response = client.execute_airflow_command(request=request)
    print(response)