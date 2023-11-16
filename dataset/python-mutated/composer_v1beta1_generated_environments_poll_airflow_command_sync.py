from google.cloud.orchestration.airflow import service_v1beta1

def sample_poll_airflow_command():
    if False:
        while True:
            i = 10
    client = service_v1beta1.EnvironmentsClient()
    request = service_v1beta1.PollAirflowCommandRequest()
    response = client.poll_airflow_command(request=request)
    print(response)